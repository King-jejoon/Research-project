"""
exp_final_m2b.py — m2 pipeline with an ASYMMETRIC dual threshold in Stage 1.

Insight from exp_final_m2: detection errors are asymmetric for the pipeline.
  early fire -> lose a little normal training data (mild)
  late fire  -> degraded cycles leak into the Stage-2 pool (severe: baseline
                contamination; e.g. seed1 unit9 fired at 54 vs true 37 and that
                seed's RUL jumped to 10.24)

So we keep TWO crossings of the SAME m2 EWMA curve:
  K_DET  = 3.0   reported transition (best detection acc 82.7%)
  K_POOL = 1.5   conservative cut for the Stage-2 training pool
                 (always <= the K=3 crossing; blocks late-fire contamination)

Everything else identical to exp_final_m2 / exp_detect_rul.
Compare: T^2 pipeline RUL 9.17 +/- 0.33 | m2 single-threshold 9.36 +/- 0.64.

Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_final_m2b.py
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from exp_detect_rul import (fit_mogp, pmean, rul_of, groups_of, true_onset,
                            SENS, SIDX, WARMUP, LAM, NPER, L0, L1, L2, INIT_THR,
                            EPOCHS, FRACS, DT)
from demo_cond2 import conditional_stats2
import neural_fusion as NF

K_DET, K_POOL = 3.0, 1.5
MCOND = 15
RUNIN_PER_UNIT = 250
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'final_m2b_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def detect_m2_dual(cache, sd):
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    X = Xall[:, SIDX]
    du = np.unique(A[:, 0].astype(int))
    ridx = []
    for u in du:
        rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
        rin = rows[cyc <= WARMUP]
        rng = np.random.default_rng(int(u) * 13 + sd)
        ridx.append(rng.choice(rin, min(RUNIN_PER_UNIT, len(rin)), replace=False))
    ridx = np.concatenate(ridx)
    gp0 = fit_mogp(W[ridx], X[ridx], m=18, steps=150)

    normal_idx = []; recs = []
    for u in du:
        rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
        ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + sd)
        idx = np.concatenate(sel)
        cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
        teX = torch.tensor(gp0['xs'].transform(W[idx]), dtype=DT)
        teY = torch.tensor(gp0['ys'].transform(X[idx]), dtype=DT)
        _, _, m2, _, _ = conditional_stats2(gp0['model'], gp0['lik'], gp0['tX'], gp0['tY'],
                                            gp0['struct'], teX, m=MCOND, test_y=teY)
        stat = np.array([m2[cc == kk].mean() for kk in ucyc])
        ic = stat[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
        base = sd0 * np.sqrt(LAM / (2 - LAM))
        e = mu0; E = []
        for v in stat: e = LAM * v + (1 - LAM) * e; E.append(e)
        E = np.array(E)

        def first_cross(K):
            cr = np.where(E > mu0 + K * base)[0]
            return int(ucyc[cr[0]]) if len(cr) else int(ucyc[-1] + 1)

        trans = first_cross(K_DET)              # reported detection
        trans_pool = first_cross(K_POOL)        # conservative pool cut (<= trans)
        trans_pool = max(trans_pool, WARMUP + 1)  # keep at least the run-in

        hs_by_cycle = np.array([A[rows[cyc == cc0], 3].mean() for cc0 in ucyc])
        tru = true_onset(hs_by_cycle, ucyc)
        pred_lab = (ucyc >= trans).astype(int); true_lab = (ucyc >= tru).astype(int)
        acc = float((pred_lab == true_lab).mean())
        recs.append(dict(unit=int(u), true=int(tru), pred=int(trans), pool=int(trans_pool),
                         delay=int(trans - tru), acc=acc))
        normal_idx.append(idx[cc < trans_pool])
    return np.concatenate(normal_idx), recs


def run_seed(sd, cache):
    torch.manual_seed(sd); np.random.seed(sd)
    normal_idx, recs = detect_m2_dual(cache, sd)

    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    Wt, Xtall, At = cache['W_test'], cache['X_s_test'], cache['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))
    rng = np.random.default_rng(sd)
    if len(normal_idx) > 5000: normal_idx = rng.choice(normal_idx, 5000, replace=False)

    gp = fit_mogp(W[normal_idx], X[normal_idx], m=18, steps=150)

    def resid(Wd, Xd, Ad, units):
        out = {}
        for u in units:
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 5 + sd)
            idx = np.concatenate(sel); cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
            pm = pmean(gp, Wd[idx]); r = Xd[idx] - pm
            d = {s: {} for s in SENS}
            for kk in np.unique(cc):
                mv = r[cc == kk].mean(0)
                for j, s in enumerate(SENS): d[s][int(kk)] = float(mv[j])
            out[int(u)] = d
        return out
    dev_res = resid(W, X, A, du); test_res = resid(Wt, Xt, At, tu)
    st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
              np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in SENS}
    nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
    dev_n, test_n = nrm(dev_res), nrm(test_res)

    def blist(dn):
        out = []
        for u in sorted(dn):
            cy = sorted(dn[u][SENS[0]].keys()); out.append((u, cy, np.array([[dn[u][s][k] for s in SENS] for k in cy])))
        return out
    dl, tl = blist(dev_n), blist(test_n)
    np.random.seed(sd)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=EPOCHS, lambda0=L0, lambda1=L1, lambda2=L2,
                            init_threshold=INIT_THR, alpha=0.001, verbose=False)
    dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
    tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
    rmse, per_f = rul_of(dHI, tHI, FRACS)
    return rmse, per_f, recs


def main():
    open(RES, 'w').close()
    cache = L.load_cache(); t0 = time.time()
    log(f'FINAL+M2-DUAL pipeline  seeds={SEEDS}  K_det={K_DET}  K_pool={K_POOL}  fracs={FRACS}')
    log(f'refs: T^2 9.17+/-0.33 | m2 single 9.36+/-0.64 '
        f'(T^2 by frac 20%:10.27 40%:5.98 60%:10.70 80%:8.83)')

    all_recs = []; rmses = []; per_fs = []
    for sd in SEEDS:
        rmse, per_f, recs = run_seed(sd, cache)
        all_recs.append(recs); rmses.append(rmse); per_fs.append(per_f)
        a = np.mean([r['acc'] for r in recs]) * 100
        pf = ' '.join(f'{f:.0%}:{v:.2f}' for f, v in per_f.items())
        log(f'  seed{sd}: det acc = {a:.1f}%  det = {[r["pred"] for r in recs]}  '
            f'pool = {[r["pool"] for r in recs]}')
        log(f'  seed{sd}: RUL RMSE = {rmse:.2f}  (by trunc {pf})  ({time.time()-t0:.0f}s)')

    flat = [r for recs in all_recs for r in recs]
    log('\n' + '=' * 74)
    log(f'DETECTION (m2, K={K_DET}): acc = {np.mean([r["acc"] for r in flat])*100:.1f}%  '
        f'mean |delay| = {np.mean([abs(r["delay"]) for r in flat]):.1f} cyc')
    log('=' * 74)
    rmses = np.array(rmses)
    log('RUL (truncation RMSE, cycles)')
    log(f'{"data used":>10} | {"seed0":>7} | {"seed1":>7} | {"seed2":>7} | {"mean":>7} | {"std":>6} | {"T2 ref":>7}')
    log('-' * 74)
    ref = {0.2: 10.27, 0.4: 5.98, 0.6: 10.70, 0.8: 8.83}
    for f in FRACS:
        vs = np.array([pf[f] for pf in per_fs])
        log(f'{f:>9.0%} | {vs[0]:>7.2f} | {vs[1]:>7.2f} | {vs[2]:>7.2f} | {vs.mean():>7.2f} | {vs.std():>6.2f} | {ref[f]:>7.2f}')
    log('-' * 74)
    log(f'{"OVERALL":>10} | {rmses[0]:>7.2f} | {rmses[1]:>7.2f} | {rmses[2]:>7.2f} | '
        f'{rmses.mean():>7.2f} | {rmses.std():>6.2f} | {9.17:>7.2f}')
    log('=' * 74)
    log(f'wall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
