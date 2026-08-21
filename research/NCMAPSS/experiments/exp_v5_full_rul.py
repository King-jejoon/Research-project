"""
exp_v5_full_rul.py — FINAL chain: the coverage-conditional gate applied to
the RUL path as well (user decision 2026-08-10: one gate everywhere).

One-time test re-summary = the NINTH DS03 test opening, disclosed.
Pre-registered: conditional per-unit V (baseline window < 10 cycles ->
V = 19.0, else V* = 20.2) on BOTH dev (tables, z-norm, HI, beta) and test;
report per-seed and pooled truncation RUL, NASA, paired p against the
fixed-gate chain.  Sanity gate: the dev conditional LOO must reproduce
7.10 +/- 0.07.  No GP refits: cached stats only.
Outputs: v5_full_rul_results.txt, v5_full_rul.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import trim25, SEEDS
from exp_v4_final import med3

VD, VS, NB = 20.2, 19.0, 10
RES = os.path.join(HERE, 'v5_full_rul_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def nb_of(dur):
    return max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))


def dev_tables(sd):
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
        V = VS if nb_of(dur) < NB else VD
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            T[i] = trim25(rs[m if m.any() else b])
        raw[u] = T; hrs[u] = np.cumsum(dur)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, raw, {u: (raw[u]-mu)/sg for u in units}, hrs, mu, sg


def test_tables(sd):
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                    if k.endswith('_cc')})
    raw = {}
    for u in units:
        cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']; dc = Zt[f'u{u}_dc']
        ucyc = Zt[f'u{u}_ucyc']; dur = Zt[f'u{u}_hours'].astype(float)
        keep = np.isin(ucyc, np.unique(cc))
        ucyc, dur = ucyc[keep], dur[keep]
        V = VS if nb_of(dur) < NB else VD
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            T[i] = trim25(rs[m if m.any() else b])
        raw[u] = T
    return units, raw


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5 FULL CHAIN — coverage-conditional gate on the RUL path, '
        'one-time test re-summary (NINTH DS03 opening, disclosed)')
    cfg = dict(HI_CFG); cfg.update(lambda1=8.0, lambda2=0.25, end_target=1.03)
    dev_rm, t_rm, t_ns = [], [], []
    Ps, Ts, Fs = [], [], []
    for sd in SEEDS:
        units, raw, Zn, hrs, mu, sg = dev_tables(sd)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        dev_rm.append(float(np.sqrt(((P-T)**2).mean())))
        t_units, t_raw = test_tables(sd)
        Zn_t = {u: (t_raw[u]-mu)/sg for u in t_units}
        HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, F2, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2-T2)**2).mean())))
        t_ns.append(nasa(P2, T2))
        Ps.append(P2); Ts.append(T2); Fs.append(F2)
        log(f'seed{sd}: dev LOO={dev_rm[-1]:.2f} shape={"OK" if ok else "FAIL"} '
            f'beta={b}  test RMSE={t_rm[-1]:.2f} NASA={t_ns[-1]:.1f}')
    log('')
    log(f'SANITY: dev conditional LOO = {np.mean(dev_rm):.2f} ± '
        f'{np.std(dev_rm):.2f}  (expected 7.10 ± 0.07)')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log(f'test RUL = {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}')
    log('test per-trunc = ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    Zm = np.load(os.path.join(HERE, 'v4_final.npz'))
    pw = st.wilcoxon(np.abs(P - T), np.abs(Zm['P'] - Zm['T']),
                     zero_method='zsplit').pvalue
    log(f'paired |err| vs fixed-gate chain (72 preds): p={pw:.3f}')
    np.savez(os.path.join(HERE, 'v5_full_rul.npz'), P=P, T=T, F=F,
             t_rm=np.array(t_rm), t_ns=np.array(t_ns),
             frac=np.array([fr[f] for f in FRACS]))
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
