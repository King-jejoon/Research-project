"""
exp_v5c_test_ablate.py — PRE-REGISTERED one-time batch = the FIFTEENTH
DS03 test opening, disclosed.  Scope frozen BEFORE the run:

  (a) HI #2 loss-term ablation on test: base / no_l0 / no_end / no_flat,
      each trained on the dev stage-2 tables only (v5c, ungated), beta by
      the frozen dev-LOO NASA rule per variant, test as prediction input.
      Sanity gates BEFORE any test file is read: the base dev side must
      reproduce 7.56 +/- 0.05; after transport the base test side must
      reproduce the thirteenth-opening 7.18 +/- 0.05.
  (b) cross-model residual-quality extension to test (18 unit-seed
      cases): llke / bspline / lr floor and SNR under the frozen metric
      of exp_v5c_resid_quality (stage-1 test healthy sigma yardstick,
      true onsets for diagnostics only); the mogp and proposed test rows
      are reprinted from the fourteenth-opening record for the combined
      view, not recomputed.

  Everything is cached-stat re-summarization; no model refits.  All
  numbers reported regardless of direction.
Outputs: v5c_test_ablate_results.txt, v5c_test_ablate.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v5c_dev import tables, L1, L2
from exp_v5c_test import test_tables
from exp_v5_ablate import train_patched
from exp_v4_hi import trim25
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, eval_units, \
    train_model_tail, shape_ok
from exp_v4_final import med3

SEEDS = [0, 1, 2]
NEND = 5
VARIANTS = ['base', 'no_l0', 'no_end', 'no_flat']
BENCH = ['llke', 'bspline', 'lr']
RES = os.path.join(HERE, 'v5c_test_ablate_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def variant_model(sd, variant):
    units, raw, Zn, hrs, mu, sg = tables(sd)
    np.random.seed(sd)
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    if variant == 'base':
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    elif variant == 'no_l0':
        cfg['lambda0'] = 0.0
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    elif variant == 'no_flat':
        cfg['flat_w'] = 0.0
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    else:                                        # no_end
        him = train_patched([Zn[u] for u in units], 0.0, cfg)
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    ok, _ = shape_ok(HIs)
    P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
    dev_rm = float(np.sqrt(((P - T) ** 2).mean()))
    return dict(him=him, mu=mu, sg=sg, HIs=HIs, hrs=hrs, b=b, units=units,
                dev_rm=dev_rm, ok=int(ok))


def quality_test(sd):
    """(b) competitor floor/SNR on the test units, frozen metric."""
    Zo = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zo.files
                    if k.endswith('_cc')})
    ons = {u: int(Zo[f'u{u}_onset'][0]) for u in units}

    def tabs_stats():
        out = {}
        for u in units:
            cc = Zo[f'u{u}_cc']; rs = Zo[f'u{u}_resid']
            ucyc = np.unique(cc)
            out[u] = (ucyc, np.stack([trim25(rs[cc == c]) for c in ucyc]))
        return out

    Zb = np.load(os.path.join(HERE, f'v4_bench_test_resid_s{sd}.npz'))
    def tabs_pool(key):
        cc, uu, rs = Zb['cc'], Zb['uu'], Zb[key]
        out = {}
        for u in units:
            m = uu == u
            ucyc = np.unique(cc[m])
            out[u] = (ucyc, np.stack([trim25(rs[m & (cc == c)])
                                      for c in ucyc]))
        return out

    mog = tabs_stats()
    pool = np.concatenate([mog[u][1][mog[u][0] < ons[u]] for u in units])
    sig_ref = pool.std(0) + 1e-12
    out = {}
    for name in BENCH:
        T = tabs_pool(f'{name}_resid')
        vals = []
        for u in units:
            uc, Tm = T[u]
            Tm = Tm / sig_ref
            h = uc < ons[u]
            fl = float(np.sqrt((Tm[h] ** 2).mean()))
            sg = float(np.abs(Tm[-NEND:]).mean())
            vals.append((fl, sg))
        out[name] = vals
    return out


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C TEST ABLATE — FIFTEENTH DS03 test opening, pre-registered: '
        '(a) HI#2 loss-term ablation on test, (b) cross-model quality '
        'extension to test')

    # dev side of every variant first — sanity gate before test contact
    models = {v: [variant_model(sd, v) for sd in SEEDS] for v in VARIANTS}
    base_dev = float(np.mean([m['dev_rm'] for m in models['base']]))
    log(f'SANITY dev base = {base_dev:.2f} (expected 7.56 ± 0.05)')
    assert abs(base_dev - 7.56) <= 0.02, 'dev sanity FAILED — stop'
    for v in VARIANTS:
        d = [m['dev_rm'] for m in models[v]]
        log(f'  dev {v:>8}: {np.mean(d):6.2f} ± {np.std(d):.2f}  '
            f'shape {sum(m["ok"] for m in models[v])}/3')

    # (a) test scoring
    log('\n(a) HI#2 loss-term variants on test')
    out, base_err = {}, None
    for v in VARIANTS:
        t_rm, t_ns, Ps, Ts, Fs = [], [], [], [], []
        for sd in SEEDS:
            m = models[v][sd]
            t_units, t_raw = test_tables(sd)
            Zn_t = {u: (t_raw[u] - m['mu']) / m['sg'] for u in t_units}
            HI_t = {u: med3(m['him'].forward(Zn_t[u]).flatten())
                    for u in t_units}
            cd = {u: np.arange(len(m['HIs'][u])) / 500.0 for u in m['units']}
            ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
            P2, T2, F2, _ = eval_units(m['b'], cd, m['HIs'], ct, HI_t, FRACS)
            t_rm.append(float(np.sqrt(((P2 - T2) ** 2).mean())))
            t_ns.append(nasa(P2, T2))
            Ps.append(P2); Ts.append(T2); Fs.append(F2)
        P = np.concatenate(Ps); T = np.concatenate(Ts)
        if v == 'base':
            log(f'  SANITY base test = {np.mean(t_rm):.2f} '
                f'(expected 7.18 ± 0.05)')
            assert abs(np.mean(t_rm) - 7.18) <= 0.02, 'test sanity FAILED'
            base_err = np.abs(P - T)
            pv = '     —'
        else:
            p = st.wilcoxon(np.abs(P - T), base_err,
                            zero_method='zsplit').pvalue
            pv = f'{p:.4f}'
        log(f'  {v:>8} | test RUL {np.mean(t_rm):6.2f} ± {np.std(t_rm):5.2f} '
            f'| NASA {np.mean(t_ns):7.1f} | p vs base {pv}')
        out[v] = dict(P=P, T=T, t_rm=np.array(t_rm), t_ns=np.array(t_ns))

    # (b) quality extension
    log('\n(b) cross-model quality on test (18 cases, stage-1 sigma '
        'yardstick)')
    acc = {}
    for sd in SEEDS:
        q = quality_test(sd)
        for name, vals in q.items():
            acc.setdefault(name, []).extend(vals)
    save_q = {}
    for name in BENCH:
        a = np.array(acc[name])
        fl, sn = a[:, 0], a[:, 1] / a[:, 0]
        log(f'  {name:>8} | floor {fl.mean():5.2f} ± {fl.std():4.2f} | '
            f'SNR {sn.mean():6.1f} ± {sn.std():4.1f}')
        save_q[f'{name}_floor'] = fl; save_q[f'{name}_snr'] = sn
    Q = np.load(os.path.join(HERE, 'v5c_resid_quality.npz'))
    fo, fc = Q['test_fl_o'], Q['test_fl_c']
    so, sc = Q['test_sg_o'] / fo, Q['test_sg_c'] / fc
    log(f'  (reprint, 14th opening)  mogp floor {fo.mean():.2f} '
        f'SNR {so.mean():.1f} | proposed floor {fc.mean():.2f} '
        f'SNR {sc.mean():.1f}')

    np.savez(os.path.join(HERE, 'v5c_test_ablate.npz'),
             **{f'{v}_{k}': val for v, d in out.items()
                for k, val in d.items()}, **save_q)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
