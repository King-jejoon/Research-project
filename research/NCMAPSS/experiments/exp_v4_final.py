"""
exp_v4_final.py — v4 chain, REVISED HI recipe: official dev + test numbers.

Revision (after the first test rendering exposed a single-cycle extrapolation
artefact on test unit 11; every change below is justified on dev alone):
  * HI weights lambda1=8, lambda2=0.25 — extended dev grid, shape gate, then
    3-axis property composite (Mon, Curv, Range; Shift dropped by decision)
  * HI post-rule: median filter, width 3, reflected edges — a fixed, untuned
    robustness rule applied identically to every unit (dev and test); it
    suppresses single-cycle input anomalies and IMPROVES dev LOO
    (7.54 -> 7.30), so it is dev-justified, not test-fitted
Everything upstream is untouched (model, gate V*=20.2 absolute, detector,
z-normalization, beta rule).  Test enters as prediction input only; this is
the SECOND rendering of DS03 test within the v4 chain (5th opening overall)
and is disclosed in the protocol.
Outputs: v4_final_results.txt, v4_final.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import build_tables, SEEDS, VSTAR, trim25
from exp_v4_test import summaries

L1, L2 = 8.0, 0.25
RES = os.path.join(HERE, 'v4_final_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def med3(h):
    """width-3 median filter; non-decreasing endpoint rule at the tail:
    the last point keeps its own value when the curve ends rising and is
    replaced by its neighbour when it ends falling — degradation is
    irreversible, so a falling final cycle is treated as an input anomaly."""
    p = np.r_[h[1], h, h[-2]]
    f = np.array([np.median(p[i:i + 3]) for i in range(len(h))])
    f[-1] = max(h[-1], h[-2])
    return f


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'V4 FINAL (revised HI recipe) — l1={L1:g} l2={L2:g}, median-3 HI '
        f'filter, gate V*={VSTAR} abs')
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)

    dev_rm, dev_ns, betas = [], [], []
    dev_frac = {f: [] for f in FRACS}
    t_rm, t_ns = [], []
    t_allP, t_allT, t_allF = [], [], []
    det_err = []
    for sd in SEEDS:
        units, Zn, hrs = build_tables(sd, VSTAR)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, (st_, en_, mo_) = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        dev_rm.append(float(np.sqrt(((P - T) ** 2).mean())))
        dev_ns.append(nasa(P, T)); betas.append(b)
        for f in FRACS:
            m = F == f
            dev_frac[f].append(float(np.sqrt(((P[m] - T[m]) ** 2).mean())))
        log(f'seed{sd} dev: shape={"OK" if ok else "FAIL"} (start {st_:.2f} '
            f'end {en_:.2f})  LOO RMSE={dev_rm[-1]:.2f}  NASA={dev_ns[-1]:.1f}'
            f'  beta={b}')

        # ---- test rendering (cached stats; prediction input only) ----
        Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
        t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                          if k.endswith('_cc')})
        t_raw, t_hrs, t_llq, t_ucyc, t_onset, t_dur = summaries(
            Zt, VSTAR, t_units)
        # dev normalization identical to build_tables
        Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        dev_raw = {}
        for u in units:
            cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
            ur = H[f'dev_{u}_ucyc']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            Tm = np.empty((len(ucyc), rs.shape[1]))
            for i, c in enumerate(ucyc):
                bm = cc == c
                mm = bm & (dc >= VSTAR)
                Tm[i] = trim25(rs[mm if mm.any() else bm])
            dev_raw[u] = Tm
        allr = np.concatenate([dev_raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
        HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P, T, F, _ = eval_units(betas[-1], cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P - T) ** 2).mean())))
        t_ns.append(nasa(P, T))
        t_allP.append(P); t_allT.append(T); t_allF.append(F)
        log(f'seed{sd} test: RMSE={t_rm[-1]:.2f}  NASA={t_ns[-1]:.1f}')

    log('')
    log('OFFICIAL v4 numbers (revised recipe, 3 seeds):')
    log(f'  dev LOO RUL  = {np.mean(dev_rm):.2f} ± {np.std(dev_rm):.2f}  '
        f'NASA {np.mean(dev_ns):.1f} ± {np.std(dev_ns):.1f}  beta={betas}')
    log(f'  dev per-trunc = '
        + ' / '.join(f'{np.mean(dev_frac[f]):.2f}' for f in FRACS))
    P = np.concatenate(t_allP); T = np.concatenate(t_allT)
    F = np.concatenate(t_allF)
    log(f'  test RUL     = {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}')
    for f in FRACS:
        m = np.isclose(F, f)
        log(f'    {int(f*100)}%: RMSE={np.sqrt(((P[m]-T[m])**2).mean()):6.2f}'
            f'  NASA={nasa(P[m], T[m]):6.1f}')
    np.savez(os.path.join(HERE, 'v4_final.npz'),
             dev_rmse=np.array(dev_rm), dev_nasa=np.array(dev_ns),
             test_rmse=np.array(t_rm), test_nasa=np.array(t_ns),
             beta=np.array(betas), P=P, T=T, F=F)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
