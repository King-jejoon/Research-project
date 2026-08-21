"""
exp_v3_test.py — v3 chain: ONE-TIME verification on the sealed DS03 test units.
(For the DS03 test split as a dataset this is the second opening — the first
was the old cycle<5 chain; recorded honestly in the paper.)

Everything below was frozen on dev BEFORE this run:
  model    : 5 sensors, RBF rank1, cycle<3, 1000/(unit,cycle) = 18k (dev only;
             training indices identical to v2_base_train_s{seed}.npz)
  gate     : detcov >= V*, V* = ABSOLUTE dev values (p15) per seed —
             no re-percentiling on test
  detector : q75 of corrected LL -> 30 flight-hour baseline -> clip
             mu0 - 10 sigma0 -> full-range mean-drop
  HI       : trim25 of gate-passing residuals, z-normalized with DEV pooled
             mu/sigma, MLP with lambda1=8, lambda2=0.5, end_target=1.03,
             others frozen (lambda0=1, thr 0.2, flat 800/0.004, 1e-3 x 1000ep),
             trained on all 9 dev units
  RUL      : cycle-axis exponential first-passage; beta selected per seed by
             the frozen dev-LOO NASA rule; truncations 20/40/60/80 %
Test data enters ONLY as prediction input.  3 seeds.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, SIDX, MCOND, DT
from demo_cond2 import conditional_stats2
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, train_model_tail
from exp_resid_clean import meandrop

SEEDS = [0, 1, 2]
NPER = 200
GATE_PCT = 15
L1, L2 = 8.0, 0.5
RES = os.path.join(HERE, 'v3_test_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def trim25(v):
    s = np.sort(v, axis=0); n = len(v)
    return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean(0)


@torch.no_grad()
def point_stats(gp, Wq, Xq, chunk=4096):
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                              gp['tY'], gp['struct'], teX,
                                              m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def build_test_stats(sd, gp):
    path = os.path.join(HERE, f'v3_test_stats_s{sd}.npz')
    if os.path.exists(path):
        return np.load(path)
    cache = L.load_cache()
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    unit = At[:, 0].astype(int); cyc = At[:, 1].astype(int); hs = At[:, 3]
    out = {}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]; hs_u = hs[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        ucyc = np.unique(cyc_u)
        idx, cc, hours = [], [], []
        for c in ucyc:
            r = rows[cyc_u == c]
            hours.append(len(r) / 3600.0)
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r); cc.append(np.full(len(r), c))
        idx = np.concatenate(idx); cc = np.concatenate(cc)
        pred, dcv, ll = point_stats(gp, Wt[idx], Xt[idx])
        hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
        below = np.where(hs_by < 0.5)[0]
        onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
        out[f'u{u}_cc'] = cc.astype(np.int32)
        out[f'u{u}_resid'] = (Xt[idx] - pred).astype(np.float32)
        out[f'u{u}_dc'] = dcv.astype(np.float32)
        out[f'u{u}_ll'] = ll.astype(np.float32)
        out[f'u{u}_ucyc'] = ucyc.astype(np.int32)
        out[f'u{u}_hours'] = np.array(hours, np.float32)
        out[f'u{u}_onset'] = np.array([onset])
    np.savez(path, **out)
    return np.load(path)


def summaries(Z, V, units, hours_key=None):
    """raw trim25 summaries + cumulative hours + q75 LL curves per unit."""
    raw, hrs, llq, ucycs, onsets, durs = {}, {}, {}, {}, {}, {}
    for u in units:
        cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']; dc = Z[f'u{u}_dc']
        ll = Z[f'u{u}_ll']; ucyc = Z[f'u{u}_ucyc']
        d = Z[f'u{u}_hours'].astype(float)
        T = np.empty((len(ucyc), rs.shape[1])); q = np.empty(len(ucyc))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            if not m.any():
                m = b
            T[i] = trim25(rs[m]); q[i] = np.percentile(ll[m], 75)
        raw[u] = T; hrs[u] = np.cumsum(d); llq[u] = q
        ucycs[u] = ucyc; onsets[u] = int(Z[f'u{u}_onset'][0]); durs[u] = d
    return raw, hrs, llq, ucycs, onsets, durs


def detect(cur, dur, ucyc):
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('V3 ONE-TIME TEST VERIFICATION (frozen chain; DS03 test 2nd opening '
        'as a dataset — recorded)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']

    det_err = []
    rul_R, rul_N = [], []
    allP, allT, allF = [], [], []
    for sd in SEEDS:
        t0 = time.time()
        z = np.load(os.path.join(HERE, f'v2_base_train_s{sd}.npz'))
        gp = fit_gp(W[z['train_idx']], X[z['train_idx']])
        log(f'seed{sd}: model rebuilt ({time.time()-t0:.0f}s)')

        # dev side (cached stats) — V* absolute from dev p15
        Zd = np.load(os.path.join(HERE, f'v2_stats_s{sd}.npz'))
        d_units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                          if k.endswith('_cc')})
        Vstar = float(np.percentile(
            np.concatenate([Zd[f'u{u}_dc'] for u in d_units]), GATE_PCT))
        # dev stats lack ucyc/hours keys; take from rul_input
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        dev_raw, dev_hrs = {}, {}
        for u in d_units:
            cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
            ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            T = np.empty((len(ucyc), rs.shape[1]))
            for i, c in enumerate(ucyc):
                b = cc == c
                m = b & (dc >= Vstar)
                T[i] = trim25(rs[m if m.any() else b])
            dev_raw[u] = T
            dev_hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc]))

        # test side
        Zt = build_test_stats(sd, gp)
        t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                          if k.endswith('_cc')})
        t_raw, t_hrs, t_llq, t_ucyc, t_onset, t_dur = summaries(
            Zt, Vstar, t_units)
        log(f'seed{sd}: test stats ready, V*={Vstar:.2f} (abs, from dev)')

        # ---- detection on test ----
        row = []
        for u in t_units:
            det = detect(t_llq[u], t_dur[u], t_ucyc[u])
            row.append((u, t_onset[u], det, det - t_onset[u]))
            det_err.append(det - t_onset[u])
        log(f'seed{sd} onset: ' + '  '.join(
            f'u{u}:t{t}/d{d_}({e:+d})' for u, t, d_, e in row))

        # ---- HI + RUL ----
        allr = np.concatenate([dev_raw[u] for u in d_units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn_d = {u: (dev_raw[u] - mu) / sg for u in d_units}
        Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
        cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn_d[u] for u in d_units], **cfg)
        HI_d = {u: him.forward(Zn_d[u]).flatten() for u in d_units}
        HI_t = {u: him.forward(Zn_t[u]).flatten() for u in t_units}
        _, _, _, beta, _ = loo('cycle', d_units, HI_d, dev_hrs, BETA_C)
        cd = {u: np.arange(len(HI_d[u])) / 500.0 for u in d_units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P, T, F, _ = eval_units(beta, cd, HI_d, ct, HI_t, FRACS)
        rm = float(np.sqrt(((P - T) ** 2).mean()))
        rul_R.append(rm); rul_N.append(nasa(P, T))
        allP.append(P); allT.append(T); allF.append(F)
        log(f'seed{sd} RUL: beta={beta}  test RMSE={rm:.2f}  '
            f'NASA={nasa(P,T):.1f}  ({time.time()-t0:.0f}s total)')

    e = np.array(det_err, float)
    log('')
    log(f'ONSET (test 6u x 3sd): RMSE={np.sqrt((e**2).mean()):.2f} cyc  '
        f'mean delay={e.mean():+.2f}  |d|<=3: {100*(np.abs(e)<=3).mean():.0f}%')
    P = np.concatenate(allP); T = np.concatenate(allT); F = np.concatenate(allF)
    log(f'RUL  (test 6u x 4fr x 3sd): RMSE={np.mean(rul_R):.2f} ± '
        f'{np.std(rul_R):.2f}   NASA={np.mean(rul_N):.1f} ± {np.std(rul_N):.1f}')
    for f in FRACS:
        m = np.isclose(F, f)
        log(f'  {int(f*100)}%: RMSE={np.sqrt(((P[m]-T[m])**2).mean()):6.2f}  '
            f'NASA={nasa(P[m], T[m]):6.1f}')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
