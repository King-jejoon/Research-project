"""
exp_v5c_bench_hr.py — PRE-REGISTERED one-time benchmark batch = the
SEVENTEENTH DS03 test opening, disclosed.  Trigger: user instruction
(2026-08-18) — retrain the Table-6 competitor normal models on the SAME
stage-2 healthy-range window as the proposed chain, so the training
window is no longer a confound in the benchmark.

Design, frozen BEFORE this run:
  window      per seed: the proposed chain's own healthy-range rows —
              v5c_stats_s{sd}.npz['train_idx'] (27k rows, cycles strictly
              before the stage-1 conditional-gate onsets), byte-identical
              to the rows the v5c MOGP trained on; the model class is the
              only variable against the proposed row.
  models      llke (h=0.1) / bspline (30 knots) / lr / cabn (lam1=0) with
              hyperparameters FROZEN at their stage-1 CV values — the same
              freeze rule the chain applied to its own GP internals
              (rbf rank1, m=18 carried into stage 2 without re-selection).
  downstream  frozen HI #2 recipe (l1=12, l2=0.25, end_target=1.03, med3),
              pooled dev z-norm per model, beta by the frozen dev-LOO NASA
              rule (BETA_C grid), UNGATED everywhere.
  report      dev LOO first (all 3 seeds, shape check — selection
              eligibility), then ONE test scoring per model: truncation
              RUL 20/40/60/80, overall RMSE, NASA; paired |err| Wilcoxon
              vs the proposed chain (v5c_test.npz) and vs the same model's
              cycle<=3 row (v5b_test_batch.npz); every number reported
              regardless of direction; unit n=6 caveat.
Sanity gates: test T vectors must be allclose to v5c_test / v5b_test_batch
before any paired test is reported.
Outputs: v5c_bench_hr_results.txt, v5c_bench_hr_resid_s{sd}.npz (dev),
         v5c_bench_hr_test_resid_s{sd}.npz, v5c_bench_hr.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_bench2_lib as B2
from exp_dsx_lib import stats_parallel
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25
from exp_v4_final import med3

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
SEEDS = [0, 1, 2]
NPER = 200
L1, L2 = 12.0, 0.25
HP = {'llke': dict(h=0.1), 'bspline': dict(n_knots=30),
      'lr': dict(), 'cabn': dict(lam1=0.0)}
MAKERS = {'llke': B2.LLKENM, 'bspline': B2.BSplineNM,
          'lr': B2.LinearNM, 'cabn': B2.CaBNNM}
RES = os.path.join(HERE, 'v5c_bench_hr_results.txt')


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
    """fit each competitor on the seed's healthy-range 27k rows and cache
    dev + test residuals on the chain sampling grid."""
    dev_p = os.path.join(HERE, f'v5c_bench_hr_resid_s{sd}.npz')
    tst_p = os.path.join(HERE, f'v5c_bench_hr_test_resid_s{sd}.npz')
    if os.path.exists(dev_p) and os.path.exists(tst_p):
        log(f'seed{sd}: residuals cached')
        return
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    Z = np.load(os.path.join(HERE, f'v5c_stats_s{sd}.npz'))
    tr = Z['train_idx']
    assert len(tr) == 27000, f'seed{sd}: unexpected train_idx size {len(tr)}'
    log(f'seed{sd}: healthy-range train rows = {len(tr)} '
        f'(onsets {dict(Z["onset_used"])})')
    d_idx, d_cc, d_uu = sample_grid(A, sd)
    t_idx, t_cc, t_uu = sample_grid(At, sd)
    dev_out = {'cc': d_cc, 'uu': d_uu}
    tst_out = {'cc': t_cc, 'uu': t_uu}
    for name, maker in MAKERS.items():
        t0 = time.time()
        m = maker(**HP[name])
        np.random.seed(0)
        m.fit(W[tr], X[tr])
        pred, _, _ = stats_parallel(m, W[d_idx], X[d_idx])
        dev_out[f'{name}_resid'] = (X[d_idx] - pred).astype(np.float32)
        pred_t, _, _ = stats_parallel(m, Wt[t_idx], Xt[t_idx])
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
    log('V5C BENCH HR — SEVENTEENTH DS03 test opening, pre-registered: '
        'competitors retrained on the proposed healthy-range window '
        '(v5c train_idx, byte-identical rows), frozen l12 downstream, '
        'ungated; hyperparameters frozen at stage-1 CV values')
    cache = L.load_cache()
    hours_dev, hours_tst = {}, {}
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        hours_dev[sd] = {int(k.split('_')[1]): H[k] for k in H.files
                         if k.startswith('dev_') and k.endswith('_hours')}
        Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
        hours_tst[sd] = {int(k[1:].split('_')[0]): Zt[k]
                         for k in Zt.files if k.endswith('_hours')}

    for sd in SEEDS:
        build_residuals(sd, cache)

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)

    # ---- dev phase first: tables, HI, shape, beta, dev LOO ----
    log('')
    log('(dev) healthy-range competitors, l12 downstream, ungated')
    side = {}
    for name in MAKERS:
        for sd in SEEDS:
            Zd = np.load(os.path.join(HERE, f'v5c_bench_hr_resid_s{sd}.npz'))
            d_raw, d_hrs = summaries(Zd['cc'], Zd['uu'],
                                     Zd[f'{name}_resid'], hours_dev[sd])
            d_units = sorted(d_raw)
            allr = np.concatenate([d_raw[u] for u in d_units])
            mu, sg = allr.mean(0), allr.std(0) + 1e-8
            Zn_d = {u: (d_raw[u] - mu) / sg for u in d_units}
            np.random.seed(sd)
            him, _ = train_model_tail([Zn_d[u] for u in d_units], **cfg)
            HI_d = {u: med3(him.forward(Zn_d[u]).flatten()) for u in d_units}
            ok, (st_, en_, mo_) = shape_ok(HI_d)
            P, T, F, beta, _ = loo('cycle', d_units, HI_d,
                                   {u: d_hrs[u] for u in d_units}, BETA_C)
            drm = float(np.sqrt(((P - T) ** 2).mean()))
            side[(name, sd)] = (him, mu, sg, HI_d, d_units, beta, drm,
                                int(ok))
            log(f'  {name} seed{sd}: dev LOO={drm:.2f} '
                f'shape={"OK" if ok else "FAIL"} '
                f'(start {st_:.2f} end {en_:.2f}) beta={beta}')

    # ---- test phase: one scoring per model ----
    log('')
    log('(test) one-time scoring, frozen from the dev side above')
    R, allpred = {}, {}
    for name in MAKERS:
        rms, nss, Ps, Ts, Fs, drms, shapes, betas = [], [], [], [], [], \
            [], [], []
        for sd in SEEDS:
            him, mu, sg, HI_d, d_units, beta, drm, ok = side[(name, sd)]
            Zt = np.load(os.path.join(HERE,
                                      f'v5c_bench_hr_test_resid_s{sd}.npz'))
            t_raw, _ = summaries(Zt['cc'], Zt['uu'],
                                 Zt[f'{name}_resid'], hours_tst[sd])
            t_units = sorted(t_raw)
            Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
            HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
            cd = {u: np.arange(len(HI_d[u])) / 500.0 for u in d_units}
            ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
            P, T, F, _ = eval_units(beta, cd, HI_d, ct, HI_t, FRACS)
            rms.append(float(np.sqrt(((P - T) ** 2).mean())))
            nss.append(nasa(P, T))
            Ps.append(P); Ts.append(T); Fs.append(F)
            drms.append(drm); shapes.append(ok); betas.append(beta)
            log(f'  {name} seed{sd}: test={rms[-1]:.2f} NASA={nss[-1]:.1f}')
        P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
        fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                                ** 2).mean())) for f in FRACS}
        R[name] = dict(rmse=(np.mean(rms), np.std(rms)),
                       nasa=(np.mean(nss), np.std(nss)),
                       dev=(np.mean(drms), np.std(drms)),
                       frac=fr, shape=int(np.sum(shapes)), beta=betas)
        allpred[name] = (P, T, F)
        log(f'  {name}: dev {np.mean(drms):.2f} ± {np.std(drms):.2f} | '
            f'test {np.mean(rms):.2f} ± {np.std(rms):.2f}  '
            f'NASA {np.mean(nss):.1f} ± {np.std(nss):.1f}  '
            f'trunc ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))

    # ---- reference rows (no recompute) + paired tests ----
    Zp = np.load(os.path.join(HERE, 'v5c_test.npz'))
    Zb = np.load(os.path.join(HERE, 'v5b_test_batch.npz'))
    log('')
    log('references (recorded, not re-run): proposed healthy-range MOGP '
        f'test {float(np.mean(Zp["t_rm"])):.2f} ± '
        f'{float(np.std(Zp["t_rm"])):.2f}; cycle<=3 rows in Table 6')
    log('')
    log('paired |err| Wilcoxon (72 preds, unit n=6 caveat):')
    for name in MAKERS:
        P, T, F = allpred[name]
        assert np.allclose(T, Zp['T']), f'{name}: pairing vs proposed off'
        p1 = st.wilcoxon(np.abs(P - T), np.abs(Zp['P'] - Zp['T']),
                         zero_method='zsplit').pvalue
        assert np.allclose(T, Zb[f'{name}_T']), f'{name}: pairing vs c3 off'
        p2 = st.wilcoxon(np.abs(P - T),
                         np.abs(Zb[f'{name}_P'] - Zb[f'{name}_T']),
                         zero_method='zsplit').pvalue
        c3 = float(np.mean(Zb[f'{name}_t_rm']))
        log(f'  {name}: vs proposed 7.18 p={p1:.3f} | '
            f'vs own cycle<=3 {c3:.2f} p={p2:.3f}')

    log('')
    log('SUMMARY — healthy-range window, frozen l12 downstream, ungated')
    log(f'  {"model":>8} | {"20%":>6} {"40%":>6} {"60%":>6} {"80%":>6} | '
        f'{"overall RMSE":>14} | {"NASA":>11} | {"dev LOO":>13} | '
        f'{"shape":>5} | beta')
    for n in MAKERS:
        r = R[n]
        log(f'  {n:>8} | ' + ' '.join(f'{r["frac"][f]:6.2f}' for f in FRACS)
            + f' | {r["rmse"][0]:6.2f} ± {r["rmse"][1]:4.2f} | '
            f'{r["nasa"][0]:5.1f} ± {r["nasa"][1]:3.1f} | '
            f'{r["dev"][0]:6.2f} ± {r["dev"][1]:4.2f} | {r["shape"]}/3 | '
            f'{r["beta"]}')
    np.savez(os.path.join(HERE, 'v5c_bench_hr.npz'),
             **{f'{n}_{k}': v for n, (P_, T_, F_) in allpred.items()
                for k, v in [('P', P_), ('T', T_), ('F', F_)]},
             **{f'{n}_t_rm': np.array([float(np.sqrt((((allpred[n][0] -
                 allpred[n][1])[i * 24:(i + 1) * 24]) ** 2).mean()))
                 for i in range(3)]) for n in MAKERS})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
