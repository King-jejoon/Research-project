"""
exp_v5c_cross_hi.py — PRE-REGISTERED mixed-stack diagnostic = the
SIXTEENTH DS03 test opening, disclosed.

Question (user, 2026-08-16): feed the STAGE-2 residuals through the
STAGE-1 health index instead of the stage-2-retrained one — what is the
test RUL?  This isolates the contribution of HI retraining from the
contribution of the residuals themselves.

Frozen design, fixed BEFORE the run:
  - HI network = HI #1 exactly as deployed: trained per seed on the
    ungated cycle<=3 dev tables, with ITS pooled z-normalization
    (mu1, sg1) kept as part of the model;
  - inputs swapped to the stage-2 residual tables (dev and test),
    normalized with (mu1, sg1);
  - beta re-derived by the frozen dev-LOO NASA rule on the resulting
    dev curves (rule constant, not a weight);
  - sanity gate BEFORE any test file is read: HI #1 on its own tables
    must reproduce dev LOO 7.15 +/- 0.10;
  - report test RUL / NASA / per-trunc / shape, paired |err| against
    the full stage-2 chain (7.18) and the full HI #1 chain (7.47),
    regardless of direction.
Outputs: v5c_cross_hi_results.txt, v5c_cross_hi.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v5c_dev import tables as s2_tables, L1, L2
from exp_v5c_test import test_tables as s2_test_tables
from exp_v4_hi import trim25
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_final import med3

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'v5c_cross_hi_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def hi1_dev(sd):
    """HI #1 dev side: ungated cycle<=3 tables + pooled z (chain code)."""
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[cc == c])
        raw[u] = T
        hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, raw, {u: (raw[u] - mu) / sg for u in units}, hrs, mu, sg


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C CROSS-HI — SIXTEENTH DS03 test opening, pre-registered: '
        'stage-2 residuals through the frozen stage-1 health index')

    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    side, dev1_rm = [], []
    for sd in SEEDS:
        u1, raw1, Zn1, hrs1, mu1, sg1 = hi1_dev(sd)
        np.random.seed(sd)
        him1, _ = train_model_tail([Zn1[u] for u in u1], **cfg)
        HIs1 = {u: med3(him1.forward(Zn1[u]).flatten()) for u in u1}
        P, T, F, b1, _ = loo('cycle', u1, HIs1, hrs1, BETA_C)
        dev1_rm.append(float(np.sqrt(((P - T) ** 2).mean())))
        side.append((him1, mu1, sg1))
        log(f'seed{sd}: HI#1 own dev LOO={dev1_rm[-1]:.2f} beta={b1}')
    log(f'SANITY HI#1 dev = {np.mean(dev1_rm):.2f} (expected 7.15 ± 0.10)')
    assert abs(np.mean(dev1_rm) - 7.15) <= 0.03, 'HI#1 sanity FAILED — stop'

    # mixed dev side: stage-2 tables through frozen HI#1
    mixed = []
    for sd in SEEDS:
        him1, mu1, sg1 = side[sd]
        u2, raw2, Zn2o, hrs2, mu2, sg2 = s2_tables(sd)
        Z2m = {u: (raw2[u] - mu1) / sg1 for u in u2}
        HIm = {u: med3(him1.forward(Z2m[u]).flatten()) for u in u2}
        ok, _ = shape_ok(HIm)
        P, T, F, bm, _ = loo('cycle', u2, HIm, hrs2, BETA_C)
        rm = float(np.sqrt(((P - T) ** 2).mean()))
        mixed.append((HIm, hrs2, bm, u2, rm, int(ok)))
        log(f'seed{sd}: MIXED dev LOO={rm:.2f} '
            f'shape={"OK" if ok else "FAIL"} beta={bm}')
    log(f'MIXED dev = {np.mean([m[4] for m in mixed]):.2f} ± '
        f'{np.std([m[4] for m in mixed]):.2f}  '
        f'shape {sum(m[5] for m in mixed)}/3')

    # test contact starts here
    t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], []
    for sd in SEEDS:
        him1, mu1, sg1 = side[sd]
        HIm, hrs2, bm, u2, _, _ = mixed[sd]
        t_units, t_raw2 = s2_test_tables(sd)
        Zt = {u: (t_raw2[u] - mu1) / sg1 for u in t_units}
        HI_t = {u: med3(him1.forward(Zt[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIm[u])) / 500.0 for u in u2}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, F2, _ = eval_units(bm, cd, HIm, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2 - T2) ** 2).mean())))
        t_ns.append(nasa(P2, T2))
        Ps.append(P2); Ts.append(T2); Fs.append(F2)
        log(f'seed{sd}: MIXED test={t_rm[-1]:.2f} NASA={t_ns[-1]:.1f}')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log('')
    log(f'MIXED STACK test RUL = {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}')
    log('test per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    C = np.load(os.path.join(HERE, 'v5c_test.npz'))
    B = np.load(os.path.join(HERE, 'v5b_test_batch.npz'))
    assert np.allclose(T, C['T']) and np.allclose(T, B['ungated_T'])
    p1 = st.wilcoxon(np.abs(P - T), np.abs(C['P'] - C['T']),
                     zero_method='zsplit').pvalue
    p2 = st.wilcoxon(np.abs(P - T), np.abs(B['ungated_P'] - B['ungated_T']),
                     zero_method='zsplit').pvalue
    log(f'paired |err| (72 preds): vs full stage-2 chain 7.18 p={p1:.3f} | '
        f'vs full HI#1 chain 7.47 p={p2:.3f}')
    np.savez(os.path.join(HERE, 'v5c_cross_hi.npz'), P=P, T=T, F=F,
             t_rm=np.array(t_rm), t_ns=np.array(t_ns),
             dev_rm=np.array([m[4] for m in mixed]))
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
