"""
exp_v4_bench_rul.py — benchmark experiment 3: each competing normal model's
residuals through the FROZEN downstream, scored on the sealed DS03 test units
exactly like the chain's own verification (exp_v4_test).

Per seed and model (llke / bspline / lr / cabn, CV-winning hyperparameters):
  dev side  : fit on the 27k training rows -> dev 9-unit full-life residuals
              (200 rows/cycle, rng u*7+sd) -> trim25 (NO gate; the competitor
              detcov is degenerate, a gate cannot be built on it) -> pooled
              dev mu/sigma -> HI net (lambda1=8, lambda2=0.25, med3, frozen)
              -> beta by the dev-LOO NASA rule -> shape check
  test side : same model -> test 6-unit full-life residuals -> dev mu/sigma
              -> dev-trained HI -> med3 -> truncation RUL at 20/40/60/80 %
mogp row: the chain's official test numbers (v4_final.npz) are reused —
no recomputation.  Protocol: this run opens DS03 test once more (7th opening,
one benchmark batch), disclosed.

Outputs: v4_bench_rul_results.txt,
         v4_bench_resid_s{sd}.npz (dev), v4_bench_test_resid_s{sd}.npz,
         per-model test RUL predictions in v4_bench_rul.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_bench2_lib as B2
from exp_v4_bench import train_idx_27k
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
SEEDS = [0, 1, 2]
NPER = 200
L1, L2 = 8.0, 0.25
HP = {'llke': dict(h=0.1), 'bspline': dict(n_knots=30),
      'lr': dict(), 'cabn': dict(lam1=0.0)}
MAKERS = {'llke': B2.LLKENM, 'bspline': B2.BSplineNM,
          'lr': B2.LinearNM, 'cabn': B2.CaBNNM}
RES = os.path.join(HERE, 'v4_bench_rul_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def sample_grid(A, sd):
    """chain sampling grid: (idx, cc, uu) for 200 rows per (unit, cycle)."""
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    idx_all, cc_all, uu_all = [], [], []
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        for c in np.unique(cyc_u):
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx_all.append(r)
            cc_all.append(np.full(len(r), c)); uu_all.append(np.full(len(r), u))
    return (np.concatenate(idx_all), np.concatenate(cc_all).astype(int),
            np.concatenate(uu_all).astype(int))


def build_residuals(sd, cache):
    """fit each competitor on the seed's 27k and cache dev+test residuals."""
    dev_p = os.path.join(HERE, f'v4_bench_resid_s{sd}.npz')
    tst_p = os.path.join(HERE, f'v4_bench_test_resid_s{sd}.npz')
    if os.path.exists(dev_p) and os.path.exists(tst_p):
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= 3)[0]
    tr = train_idx_27k(sd, unit, cyc, pool)
    d_idx, d_cc, d_uu = sample_grid(A, sd)
    t_idx, t_cc, t_uu = sample_grid(At, sd)
    dev_out = {'cc': d_cc, 'uu': d_uu}
    tst_out = {'cc': t_cc, 'uu': t_uu}
    # seed-0 dev residuals already exist in v4_bench_resid.npz — reuse
    old = None
    if sd == 0 and os.path.exists(os.path.join(HERE, 'v4_bench_resid.npz')):
        old = np.load(os.path.join(HERE, 'v4_bench_resid.npz'))
    for name, maker in MAKERS.items():
        t0 = time.time()
        m = maker(**HP[name])
        np.random.seed(0)
        m.fit(W[tr], X[tr])
        if old is not None and f'{name}_resid' in old.files:
            dev_out[f'{name}_resid'] = old[f'{name}_resid']
        else:
            pred, _, _ = m.stats(W[d_idx], X[d_idx], chunk=4096)
            dev_out[f'{name}_resid'] = (X[d_idx] - pred).astype(np.float32)
        pred_t, _, _ = m.stats(Wt[t_idx], Xt[t_idx], chunk=4096)
        tst_out[f'{name}_resid'] = (Xt[t_idx] - pred_t).astype(np.float32)
        log(f'seed{sd} {name}: residuals ready ({time.time()-t0:.0f}s)')
    np.savez(dev_p, **dev_out)
    np.savez(tst_p, **tst_out)


def summaries(cc, uu, rs, hours_map):
    """per-unit trim25 tables (no gate) + cumulative hours."""
    raw, hrs = {}, {}
    for u in np.unique(uu):
        m = uu == u
        ucyc = np.unique(cc[m])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[m & (cc == c)])
        raw[int(u)] = T
        hrs[int(u)] = np.cumsum(hours_map[int(u)][:len(ucyc)])
    return raw, hrs


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('BENCH EXPERIMENT 3 — competitor residuals through the frozen '
        'downstream, scored on DS03 test (7th opening, benchmark batch)')
    cache = L.load_cache()
    # flight hours per cycle for dev (beta rule needs hrs; cycle axis used)
    hours_dev = {}
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        hours_dev[sd] = {int(k.split('_')[1]): H[k] for k in H.files
                         if k.startswith('dev_') and k.endswith('_hours')}
    # test hours from chain test stats (independent of model)
    hours_tst = {}
    for sd in SEEDS:
        Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
        hours_tst[sd] = {int(k[1:].split('_')[0]): Zt[k]
                         for k in Zt.files if k.endswith('_hours')}

    for sd in SEEDS:
        build_residuals(sd, cache)

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    R = {}
    allpred = {}
    for name in MAKERS:
        rms, nss, betas, shapes = [], [], [], []
        Ps, Ts, Fs = [], [], []
        for sd in SEEDS:
            Zd = np.load(os.path.join(HERE, f'v4_bench_resid_s{sd}.npz'))
            Zt = np.load(os.path.join(HERE, f'v4_bench_test_resid_s{sd}.npz'))
            d_raw, d_hrs = summaries(Zd['cc'], Zd['uu'],
                                     Zd[f'{name}_resid'], hours_dev[sd])
            t_raw, _ = summaries(Zt['cc'], Zt['uu'],
                                 Zt[f'{name}_resid'], hours_tst[sd])
            d_units = sorted(d_raw); t_units = sorted(t_raw)
            allr = np.concatenate([d_raw[u] for u in d_units])
            mu, sg = allr.mean(0), allr.std(0) + 1e-8
            Zn_d = {u: (d_raw[u] - mu) / sg for u in d_units}
            Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
            np.random.seed(sd)
            him, _ = train_model_tail([Zn_d[u] for u in d_units], **cfg)
            HI_d = {u: med3(him.forward(Zn_d[u]).flatten()) for u in d_units}
            HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
            ok, (st_, en_, mo_) = shape_ok(HI_d)
            shapes.append(int(ok))
            _, _, _, beta, _ = loo('cycle', d_units, HI_d,
                                   {u: d_hrs[u] for u in d_units}, BETA_C)
            betas.append(beta)
            cd = {u: np.arange(len(HI_d[u])) / 500.0 for u in d_units}
            ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
            P, T, F, _ = eval_units(beta, cd, HI_d, ct, HI_t, FRACS)
            rms.append(float(np.sqrt(((P - T) ** 2).mean())))
            nss.append(nasa(P, T))
            Ps.append(P); Ts.append(T); Fs.append(F)
            log(f'{name} seed{sd}: shape={"OK" if ok else "FAIL"} '
                f'(start {st_:.2f} end {en_:.2f}) beta={beta} '
                f'test RMSE={rms[-1]:.2f} NASA={nss[-1]:.1f}')
        P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
        fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                                ** 2).mean())) for f in FRACS}
        R[name] = dict(rmse=(np.mean(rms), np.std(rms)),
                       nasa=(np.mean(nss), np.std(nss)),
                       frac=fr, shape=int(np.sum(shapes)), beta=betas)
        allpred[name] = (P, T, F)

    # mogp official row
    Zm = np.load(os.path.join(HERE, 'v4_final.npz'))
    Pm, Tm, Fm = Zm['P'], Zm['T'], Zm['F']
    frm = {f: float(np.sqrt(((Pm[np.isclose(Fm, f)] - Tm[np.isclose(Fm, f)])
                             ** 2).mean())) for f in FRACS}
    R['mogp'] = dict(rmse=(float(Zm['test_rmse'].mean()),
                           float(Zm['test_rmse'].std())),
                     nasa=(float(Zm['test_nasa'].mean()),
                           float(Zm['test_nasa'].std())),
                     frac=frm, shape=3, beta=list(Zm['beta']))
    allpred['mogp'] = (Pm, Tm, Fm)

    log('')
    log('SUMMARY — test truncation RUL, frozen downstream (mogp row = the '
        'chain with its gate; competitors ungated, gate not constructible)')
    log(f'  {"model":>8} | {"20%":>6} {"40%":>6} {"60%":>6} {"80%":>6} | '
        f'{"overall RMSE":>14} | {"NASA":>11} | {"shape":>5} | beta')
    for n in ['mogp', 'llke', 'bspline', 'lr', 'cabn']:
        r = R[n]
        log(f'  {n:>8} | ' + ' '.join(f'{r["frac"][f]:6.2f}' for f in FRACS)
            + f' | {r["rmse"][0]:6.2f} ± {r["rmse"][1]:4.2f} | '
            f'{r["nasa"][0]:5.1f} ± {r["nasa"][1]:3.1f} | {r["shape"]}/3 | '
            f'{r["beta"]}')
    np.savez(os.path.join(HERE, 'v4_bench_rul.npz'),
             **{f'{n}_{k}': v for n, (P_, T_, F_) in allpred.items()
                for k, v in [('P', P_), ('T', T_), ('F', F_)]})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
