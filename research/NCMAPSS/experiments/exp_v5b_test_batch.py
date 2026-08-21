"""
exp_v5b_test_batch.py — PRE-REGISTERED one-time test batch = the TENTH
DS03 test opening, disclosed.  Trigger: chain re-freeze on the conditional-
table lambda re-sweep winner (lambda1 = 12, lambda2 = 0.25; user decision
2026-08-10).

Pre-registration, frozen BEFORE this run (dev side fully derived in
exp_v5b_dev / v5b_acb):
  (a) re-frozen chain (conditional gate, l12, med3, beta per seed from the
      dev-LOO NASA rule = [30, 30, 35]) -> test truncation RUL, NASA,
      paired p against the lambda8 conditional chain (v5_full_rul.npz);
  (b) MOGP ungated, l12 downstream -> test RUL;
  (c) competitors (llke / bspline / lr / cabn, ungated), l12 downstream,
      cached residuals -> test RUL.
Everything is cached-stat re-summarization; no model refits.  Report all
numbers regardless of direction.  Sanity gate: the dev side must reproduce
v5b_dev (7.22 +/- 0.08) before test files are touched.
Outputs: v5b_test_batch_results.txt, v5b_test_batch.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_hi import build_tables, trim25, SEEDS
from exp_v4_final import med3
from exp_v5_full_rul import dev_tables, test_tables

L1, L2 = 12.0, 0.25
VOFF = -1e9
BENCH = ['llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'v5b_test_batch_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def dev_ungated(sd):
    units, Zn, hrs = build_tables(sd, VOFF)
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    raw = {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']
        ur = H[f'dev_{u}_ucyc']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[cc == c])
        raw[u] = T
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, raw, {u: (raw[u]-mu)/sg for u in units}, hrs, mu, sg


def test_ungated(sd):
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                    if k.endswith('_cc')})
    raw = {}
    for u in units:
        cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']
        ucyc = Zt[f'u{u}_ucyc']
        keep = np.isin(ucyc, np.unique(cc))
        ucyc = ucyc[keep]
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[cc == c])
        raw[u] = T
    return units, raw


def bench_dev(sd, name):
    Zd = np.load(os.path.join(HERE, f'v4_bench_resid_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    hours = {int(k.split('_')[1]): H[k] for k in H.files
             if k.startswith('dev_') and k.endswith('_hours')}
    cc, uu, rs = Zd['cc'], Zd['uu'], Zd[f'{name}_resid']
    raw, hrs = {}, {}
    for u in np.unique(uu):
        m = uu == u
        ucyc = np.unique(cc[m])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[m & (cc == c)])
        raw[int(u)] = T
        hrs[int(u)] = np.cumsum(hours[int(u)][:len(ucyc)])
    units = sorted(raw)
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    return units, raw, {u: (raw[u]-mu)/sg for u in units}, hrs, mu, sg


def bench_test(sd, name):
    Zd = np.load(os.path.join(HERE, f'v4_bench_test_resid_s{sd}.npz'))
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    cc, uu, rs = Zd['cc'], Zd['uu'], Zd[f'{name}_resid']
    raw = {}
    for u in np.unique(uu):
        m = uu == u
        ucyc = np.unique(cc[m])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            T[i] = trim25(rs[m & (cc == c)])
        raw[int(u)] = T
    return sorted(raw), raw


def run_config(tag, dev_loader, test_loader):
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    dev_rm, t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], [], []
    for sd in SEEDS:
        units, raw, Zn, hrs, mu, sg = dev_loader(sd)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        dev_rm.append(float(np.sqrt(((P-T)**2).mean())))
        t_units, t_raw = test_loader(sd)
        Zn_t = {u: (t_raw[u]-mu)/sg for u in t_units}
        HI_t = {u: med3(him.forward(Zn_t[u]).flatten()) for u in t_units}
        cd = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P2, T2, F2, _ = eval_units(b, cd, HIs, ct, HI_t, FRACS)
        t_rm.append(float(np.sqrt(((P2-T2)**2).mean())))
        t_ns.append(nasa(P2, T2))
        Ps.append(P2); Ts.append(T2); Fs.append(F2)
        log(f'  {tag} seed{sd}: dev LOO={dev_rm[-1]:.2f} '
            f'shape={"OK" if ok else "FAIL"} beta={b} '
            f'test={t_rm[-1]:.2f} NASA={t_ns[-1]:.1f}')
    P = np.concatenate(Ps); T = np.concatenate(Ts); F = np.concatenate(Fs)
    fr = {f: float(np.sqrt(((P[np.isclose(F, f)] - T[np.isclose(F, f)])
                            ** 2).mean())) for f in FRACS}
    log(f'  {tag}: dev {np.mean(dev_rm):.2f} ± {np.std(dev_rm):.2f} | '
        f'test {np.mean(t_rm):.2f} ± {np.std(t_rm):.2f}  '
        f'NASA {np.mean(t_ns):.1f} ± {np.std(t_ns):.1f}  '
        f'trunc ' + ' / '.join(f'{fr[f]:.2f}' for f in FRACS))
    return dict(P=P, T=T, F=F, t_rm=t_rm, t_ns=t_ns,
                fr=[fr[f] for f in FRACS], dev_rm=dev_rm)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B TEST BATCH — TENTH DS03 test opening, pre-registered, '
        're-frozen chain l1=12 l2=0.25')
    out = {}
    log('\n(a) re-frozen chain, conditional gate')
    out['chain'] = run_config('chain', dev_tables, test_tables)
    sanity = np.mean(out['chain']['dev_rm'])
    log(f'  SANITY dev = {sanity:.2f} (expected 7.22 ± 0.08)')
    Z8 = np.load(os.path.join(HERE, 'v5_full_rul.npz'))
    pw = st.wilcoxon(np.abs(out['chain']['P'] - out['chain']['T']),
                     np.abs(Z8['P'] - Z8['T']), zero_method='zsplit').pvalue
    log(f'  paired |err| vs lambda8 conditional chain (72 preds): p={pw:.3f}')

    log('\n(b) MOGP ungated, l12 downstream')
    out['ungated'] = run_config('ungated', dev_ungated, test_ungated)

    log('\n(c) competitors, l12 downstream, ungated')
    for n in BENCH:
        out[n] = run_config(n, lambda sd, n=n: bench_dev(sd, n),
                            lambda sd, n=n: bench_test(sd, n))

    np.savez(os.path.join(HERE, 'v5b_test_batch.npz'),
             **{f'{k}_{p}': np.asarray(v) for k in out
                for p, v in out[k].items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
