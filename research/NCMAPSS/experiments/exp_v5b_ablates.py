"""
exp_v5b_ablates.py — ablation suite re-based on the RE-FROZEN chain
(lambda1 = 12, lambda2 = 0.25, conditional gate everywhere).  Dev only.

1. HI loss-term ablation on conditional tables (base/no_l0/no_end/no_flat)
   — machinery reused from exp_v5_ablate with L1 patched to 12.
2. Gate vs ungated on the final metric (RUL paired p; onset is lambda-free
   and unchanged: conditional 5.21 vs ungated 6.78, p = 0.003).
3. Raw sensor means replace residuals (protocol = exp_v4_ablate2raw, l12).
4. Healthy-range window retraining (exp_v4_exp2 cached stats, gate off, l12).
Outputs: v5b_ablates_results.txt
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_v5_ablate as A5                   # loss-term machinery
A5.L1, A5.L2 = 12.0, 0.25                    # re-frozen recipe
import exp_v4_exp2 as E2
E2.L1, E2.L2 = 12.0, 0.25
import exp_lib as L
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_final import med3
from exp_v4_hi import SEEDS

RES = os.path.join(HERE, 'v5b_ablates_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def raw_sensor(cfg):
    SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
    SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
    NPER = 200
    cache = L.load_cache()
    X, A = cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    rms, nss, oks = [], [], []
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        raw, hrs = {}, {}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cyc_u) if int(c) in pos])
            T = np.empty((len(ucyc), len(SENS)))
            for i, c in enumerate(ucyc):
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                T[i] = X[r].mean(0)
            raw[int(u)] = T
            hrs[int(u)] = np.cumsum(
                np.array([durs[pos[int(c)]] for c in ucyc], float))
        units = sorted(raw)
        allr = np.concatenate([raw[u] for u in units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn = {u: (raw[u] - mu) / sg for u in units}
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok))
    return rms, nss, oks


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B ABLATIONS — re-frozen chain lambda1=12, lambda2=0.25, dev only')

    log('\n1. HI loss-term ablation (conditional tables, l12)')
    log(f'  {"variant":>8} | {"RUL RMSE":>13} | {"NASA":>8} | shape | start/end')
    base_err = None
    for v in ['base', 'no_l0', 'no_end', 'no_flat']:
        rms, nss, oks, sts, ens, errs = [], [], [], [], [], []
        for sd in SEEDS:
            r, n_, ok, s_, e_, e = A5.rul_variant(sd, v)
            rms.append(r); nss.append(n_); oks.append(ok)
            sts.append(s_); ens.append(e_); errs.append(e)
        if v == 'base':
            base_err = np.concatenate(errs)
        log(f'  {v:>8} | {np.mean(rms):6.2f} ± {np.std(rms):4.2f} | '
            f'{np.mean(nss):8.1f} | {int(np.sum(oks))}/3 | '
            f'{np.mean(sts):.2f}/{np.mean(ens):.2f}')

    log('\n2. Gate vs ungated on the final metric (l12)')
    ung_errs = []
    for sd in SEEDS:
        r, n_, ok, s_, e_, e = A5.rul_variant(sd, 'base', gated=False)
        ung_errs.append(e)
        log(f'  ungated seed{sd}: RUL={r:.2f}')
    ung = np.concatenate(ung_errs)
    pw = st.wilcoxon(np.abs(base_err), np.abs(ung),
                     zero_method='zsplit').pvalue
    log(f'  RUL: conditional {np.sqrt((base_err**2).mean()):.2f} vs ungated '
        f'{np.sqrt((ung**2).mean()):.2f}, paired p={pw:.3f}')
    log('  onset (lambda-free, from exp_v5_ablate): conditional 5.21 vs '
        'ungated 6.78, p=0.003')

    log('\n3. Raw sensor means replace residuals (l12)')
    cfg = dict(HI_CFG); cfg.update(lambda1=12.0, lambda2=0.25,
                                   end_target=1.03)
    rms, nss, oks = raw_sensor(cfg)
    log(f'  RAW-MEAN: RUL {np.mean(rms):.2f} ± {np.std(rms):.2f}  '
        f'NASA {np.mean(nss):.1f}  shape {int(np.sum(oks))}/3')

    log('\n4. Healthy-range window retraining (cached exp2 stats, gate off, '
        'l12)')
    r0 = E2.rul_at(-1e9)
    log(f'  healthy-range: RUL {r0[0]:.2f} ± {r0[1]:.2f}  NASA {r0[2]:.1f}  '
        f'shape {r0[3]}/3')

    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
