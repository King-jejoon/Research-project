"""
exp_gp_selfdrop.py — Does detcov-based TRAINING-SET selection improve the
normal GP model itself (Stage 1), the way the Stage-3b residual filter does?

Protocol per seed:
  0. tr    = rng.choice(cycle<5 pool, 8192)      (identical draw to baseline cache)
     cand  = 32768 fresh pool points (disjoint from tr)
     held  = 4096  fresh pool points (disjoint)  -> GP quality diagnostics
  1. gp0   = fit on tr (baseline GP)
  2. score cand with gp0 -> detcov per point
  3. for each gpdrop level: within every dev unit-cycle group of cand, drop the
     bottom gpdrop fraction by detcov (keep confident-region points), resample
     8192 from survivors, refit GP, recompute per-point stats for ALL units
     (NPER=200) -> gpdrop_stats_g<pct>_s<sd>.npz
  4. downstream identical to the FINAL system: per-cycle causal detcov filter
     (drop 0/60/85%), HI = train_model_tail lambda=(1,1,2,0.25)+L_flat(300,.002),
     Bayesian first-passage truncation RUL (20/40/60/80%), 3 seeds.
Control (gpdrop=0) reuses port_stats_s3_matern32_r1_m18_n8192_p200_s<sd>.npz —
distributionally the same GP training draw, identical downstream.

GP diagnostics per variant: held-out normal residual RMSE per sensor,
held-out mean m2 (~chi2(3)=3 in control), EWMA detection accuracy.

Run:   /opt/anaconda3/envs/pt_prac/bin/python3 exp_gp_selfdrop.py
Smoke: /opt/anaconda3/envs/pt_prac/bin/python3 exp_gp_selfdrop.py --smoke
"""
import os, sys, time, argparse
import numpy as np
import torch
torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from exp_ds03_port import fit_mogp, groups_of, rul_eval, nasa_score, true_onset, FRACS, DT
from demo_cond2 import conditional_stats2
from neural_fusion_tail import train_model_tail

SENS = ['T48', 'T50', 'Wf']
SEEDS = [0, 1, 2]
GPDROPS = [0.50, 0.25, 0.75]          # run order: biggest expected signal first
DS_DROPS = [0.0, 0.60, 0.85]          # downstream stage-3b filter levels
NPER, WARMUP, LAM = 200, 10, 0.2
NPTS, NCAND, NHELD = 8192, 32768, 4096
KERNEL, RANK, M, STEPS, LR, MCOND = 'matern32', 1, 18, 120, 0.05, 15
HI_KW = dict(epochs=1000, lambda0=1.0, lambda1=2.0, lambda2=0.25,
             init_threshold=0.2, flat_w=300.0, flat_m=0.002, alpha=0.001)
CTRL_CACHE = 'port_stats_s3_matern32_r1_m18_n8192_p200_s{sd}.npz'
RES = os.path.join(HERE, 'gp_selfdrop_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def score_points(gp, Wq, Xq, chunk=8192):
    """conditional stats for query points, chunked. Returns detcov, m2, resid."""
    dcs, m2s, rss = [], [], []
    for a in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[a:a+chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[a:a+chunk]), dtype=DT)
        pred_s, dc, m2, _, _ = conditional_stats2(
            gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'], teX,
            m=MCOND, test_y=teY)
        pred = pred_s * gp['ys'].scale_ + gp['ys'].mean_
        dcs.append(dc); m2s.append(m2); rss.append(Xq[a:a+chunk] - pred)
    return np.concatenate(dcs), np.concatenate(m2s), np.concatenate(rss)


def full_stats(gp, cache, sidx, sd, path):
    """per-point stats for every dev+test unit at NPER (mirrors exp_ds03_port)."""
    if os.path.exists(path):
        return np.load(path, allow_pickle=False)
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    Wt, Xtall, At = cache['W_test'], cache['X_s_test'], cache['A_test']
    X, Xt = Xall[:, sidx], Xtall[:, sidx]
    out = {}
    for split, (Wd, Xd, Ad) in [('dev', (W, X, A)), ('test', (Wt, Xt, At))]:
        for u in np.unique(Ad[:, 0].astype(int)):
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + sd)
            idx = np.concatenate(sel)
            cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
            dc, m2, resid = score_points(gp, Wd[idx], Xd[idx])
            key = f'{split}_{u}'
            out[f'{key}_cc'] = cc.astype(np.int32)
            out[f'{key}_resid'] = resid.astype(np.float32)
            out[f'{key}_detcov'] = dc.astype(np.float32)
            out[f'{key}_m2'] = m2.astype(np.float32)
            if split == 'dev':
                hs = np.array([Ad[rows[cyc == c0], 3].mean() for c0 in ucyc])
                out[f'{key}_hs'] = hs.astype(np.float32)
    np.savez_compressed(path, **out)
    return np.load(path, allow_pickle=False)


def detect_acc(z):
    """EWMA(K=3) on per-cycle mean m2, scored vs hs, all dev units."""
    accs, delays = [], []
    du = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('dev_') and k.endswith('_cc')})
    for u in du:
        key = f'dev_{u}'; cc = z[f'{key}_cc']; m2 = z[f'{key}_m2']
        ucyc = np.unique(cc)
        m2c = np.array([m2[cc == k].mean() for k in ucyc])
        tru = true_onset(z[f'{key}_hs'], ucyc)
        ic = m2c[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
        UCL = mu0 + 3.0 * sd0 * np.sqrt(LAM / (2 - LAM))
        e = mu0; E = []
        for v in m2c: e = LAM * v + (1 - LAM) * e; E.append(e)
        cr = np.where(np.array(E) > UCL)[0]
        trans = int(ucyc[cr[0]]) if len(cr) else int(ucyc[-1] + 1)
        accs.append(float(((ucyc >= trans).astype(int) == (ucyc >= tru).astype(int)).mean()))
        delays.append(abs(trans - tru))
    return np.mean(accs) * 100, np.mean(delays)


def build_resid(z, drop):
    """residual cycle-means with per-cycle CAUSAL detcov threshold."""
    def rd(split):
        units = sorted({int(k.split('_')[1]) for k in z.files
                        if k.startswith(f'{split}_') and k.endswith('_cc')})
        out = {}
        for u in units:
            key = f'{split}_{u}'; cc = z[f'{key}_cc']; r = z[f'{key}_resid']; dc = z[f'{key}_detcov']
            d = {s: {} for s in SENS}
            for k in np.unique(cc):
                m = cc == k
                if drop > 0:
                    thr = np.percentile(dc[m], drop * 100)
                    keep = m & (dc >= thr)
                    if keep.sum() == 0: keep = m
                else:
                    keep = m
                mv = r[keep].mean(0)
                for j, s in enumerate(SENS): d[s][int(k)] = float(mv[j])
            out[u] = d
        return out
    dev, test = rd('dev'), rd('test')
    st = {s: (np.mean([v for u in dev for v in dev[u][s].values()]),
              np.std([v for u in dev for v in dev[u][s].values()]) + 1e-8) for s in SENS}
    nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
    dn, tn = nrm(dev), nrm(test)
    bl = lambda dd: [(u, sorted(dd[u][SENS[0]].keys()),
                      np.array([[dd[u][s][k] for s in SENS] for k in sorted(dd[u][SENS[0]].keys())]))
                     for u in sorted(dd)]
    return bl(dn), bl(tn)


def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def shape_of(h):
    hs = smooth1d(h); n = len(hs); x = np.linspace(0, 1, n)
    start = hs[x <= 0.10].mean() if (x <= 0.10).any() else hs[0]
    s9m = (x >= 0.8) & (x <= 0.9); s10m = (x >= 0.9)
    s9 = np.polyfit(x[s9m], hs[s9m], 1)[0]; s10 = np.polyfit(x[s10m], hs[s10m], 1)[0]
    return start, s9, s10, hs[-1]


def downstream(zs, hi_kw):
    """zs: {sd: npz}. Full HI+RUL at each DS_DROPS level, 3 seeds."""
    rows = []
    for dsd in DS_DROPS:
        rmses, scores, mets = [], [], []
        for sd in SEEDS:
            dl, tl = build_resid(zs[sd], dsd)
            np.random.seed(sd)
            him, _ = train_model_tail([d for _, _, d in dl], **hi_kw)
            dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
            tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
            P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
            rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
            scores.append(nasa_score(P, T))
            for u in tHI: mets.append(shape_of(tHI[u]))
        start, s9, s10, end = np.nanmean(np.array(mets), 0)
        ok = (s10 >= s9) and (start <= 0.30) and (end >= 0.93)
        rows.append((dsd, np.mean(rmses), np.std(rmses), np.mean(scores), s9, s10, ok))
    return rows


def prep_variant(gpd, sd, cache, sidx, cfgsizes, res_diag):
    """fit stage-0 GP, score candidates, filter, refit, full stats. cached."""
    npts, ncand, nheld, steps, epochs = cfgsizes
    path = os.path.join(HERE, f'gpdrop_stats_g{int(gpd*100)}_s{sd}'
                              + ('_smoke' if npts != NPTS else '') + '.npz')
    if os.path.exists(path):
        return np.load(path, allow_pickle=False), None
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    X = Xall[:, sidx]
    cycles = A[:, 1].astype(int)
    pool = np.where(cycles < 5)[0]
    rng = np.random.default_rng(sd)
    tr = rng.choice(pool, min(npts, len(pool)), replace=False)   # == baseline draw
    rest = np.setdiff1d(pool, tr)
    cand = rng.choice(rest, min(ncand, len(rest)), replace=False)
    rest2 = np.setdiff1d(rest, cand)
    held = rng.choice(rest2, min(nheld, len(rest2)), replace=False)

    t0 = time.time()
    torch.manual_seed(sd)
    gp0 = fit_mogp(W[tr], X[tr], KERNEL, RANK, M, steps, lr=LR)
    dc_c, m2_c, _ = score_points(gp0, W[cand], X[cand])

    # per unit-cycle group percentile filter on candidate detcov
    grp = A[cand, 0].astype(int) * 100 + A[cand, 1].astype(int)
    keep = np.zeros(len(cand), bool)
    for g in np.unique(grp):
        m = grp == g
        thr = np.percentile(dc_c[m], gpd * 100)
        keep |= m & (dc_c >= thr)
    surv = cand[keep]
    rng2 = np.random.default_rng(sd * 1000 + int(gpd * 100))
    tr2 = rng2.choice(surv, min(npts, len(surv)), replace=False)

    torch.manual_seed(sd)
    gp2 = fit_mogp(W[tr2], X[tr2], KERNEL, RANK, M, steps, lr=LR)
    # held-out diagnostics for both GPs
    _, m2_h0, rs_h0 = score_points(gp0, W[held], X[held])
    _, m2_h2, rs_h2 = score_points(gp2, W[held], X[held])
    d0 = np.sqrt((rs_h0 ** 2).mean(0)); d2 = np.sqrt((rs_h2 ** 2).mean(0))
    diag = (f'  g{int(gpd*100)} s{sd}: heldout resid RMSE {SENS} '
            f'base={np.round(d0,3).tolist()} filt={np.round(d2,3).tolist()}  '
            f'm2 base={m2_h0.mean():.2f} filt={m2_h2.mean():.2f}  '
            f'surv={len(surv)}  ({time.time()-t0:.0f}s)')
    res_diag.append(diag); log(diag)
    z = full_stats(gp2, cache, sidx, sd, path)
    log(f'  g{int(gpd*100)} s{sd}: full stats ready ({time.time()-t0:.0f}s total)')
    return z, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--gpdrops', default=None, help='comma list, e.g. 0.5,0.25')
    args = ap.parse_args()

    global SEEDS, NPER
    gpdrops = [float(x) for x in args.gpdrops.split(',')] if args.gpdrops else GPDROPS
    if args.smoke:
        SEEDS = [0]; NPER = 20
        sizes = (512, 2048, 512, 20, 50)
        gpdrops = [0.5]
    else:
        sizes = (NPTS, NCAND, NHELD, STEPS, HI_KW['epochs'])
    hi_kw = dict(HI_KW); hi_kw['epochs'] = sizes[4]

    open(RES, 'a').close()
    log(f'\n===== GP SELF-DROP experiment  {time.strftime("%m-%d %H:%M")} '
        f'gpdrops={gpdrops} smoke={args.smoke} =====')
    log(f'  GP: s3 {KERNEL} r{RANK} m{M} npts={sizes[0]} steps={sizes[3]} lr={LR} | '
        f'cand={sizes[1]} held={sizes[2]} NPER={NPER} | HI final(L_flat) | ds_drops={DS_DROPS}')
    cache = L.load_cache()
    sidx = [L.OUTPUT_NAMES.index(s) for s in SENS]

    # ---- control: existing baseline caches through identical downstream ----
    if not args.smoke:
        zs = {sd: np.load(os.path.join(HERE, CTRL_CACHE.format(sd=sd))) for sd in SEEDS}
        a, dly = np.mean([detect_acc(zs[sd]) for sd in SEEDS], 0)
        log(f'\n[control gpdrop=0]  det acc={a:.1f}%  |delay|={dly:.1f}')
        for dsd, r, rs, sc, s9, s10, ok in downstream(zs, hi_kw):
            log(f'  ds_drop={dsd:.0%}: RMSE {r:.2f} ±{rs:.2f}  score {sc:.1f}  '
                f's9->s10 {s9:.2f}->{s10:.2f}  {"PASS" if ok else "fail"}')

    # ---- variants ----
    for gpd in gpdrops:
        diags = []
        zs = {}
        for sd in SEEDS:
            zs[sd], _ = prep_variant(gpd, sd, cache, sidx, sizes, diags)
        a, dly = np.mean([detect_acc(zs[sd]) for sd in SEEDS], 0)
        log(f'\n[gpdrop={gpd:.0%}]  det acc={a:.1f}%  |delay|={dly:.1f}')
        for dsd, r, rs, sc, s9, s10, ok in downstream(zs, hi_kw):
            log(f'  ds_drop={dsd:.0%}: RMSE {r:.2f} ±{rs:.2f}  score {sc:.1f}  '
                f's9->s10 {s9:.2f}->{s10:.2f}  {"PASS" if ok else "fail"}')
    log('\ndone.')


if __name__ == '__main__':
    main()
