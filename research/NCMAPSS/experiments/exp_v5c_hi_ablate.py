"""
exp_v5c_hi_ablate.py — HI #2 loss-term ablation on the PROPOSED stage-2
base (v5c healthy-range tables, ungated), DEV ONLY:
base / no_l0 / no_end / no_flat.

Machinery reused from exp_v5_ablate (the end term is removed via the
source-patched end_w switch, as in v4_ablate13); tables come from
exp_v5c_dev.tables (ungated, healthy-range stats).  Sanity: base must
reproduce v5c_dev (7.56 +/- 0.05).  Test scoring is NOT done here — it
is the planned pre-registered FIFTEENTH opening, run separately.
Outputs: v5c_hi_ablate_results.txt, v5c_hi_ablate.npz
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v5c_dev import tables, L1, L2
from exp_v5_ablate import train_patched
from exp_rul_r23 import HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok
from exp_v4_final import med3

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'v5c_hi_ablate_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def run_variant(sd, variant):
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
    elif variant == 'no_end':
        him = train_patched([Zn[u] for u in units], 0.0, cfg)
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    ok, _ = shape_ok(HIs)
    P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
    return float(np.sqrt(((P - T) ** 2).mean())), nasa(P, T), int(ok), P - T


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5C HI#2 LOSS-TERM ABLATION — stage-2 base (healthy-range tables, '
        'ungated), dev only; test = planned 15th opening, not run here')
    out, base_err = {}, None
    for v in ['base', 'no_l0', 'no_end', 'no_flat']:
        rms, nss, oks, errs = [], [], [], []
        for sd in SEEDS:
            r, n_, ok, e = run_variant(sd, v)
            rms.append(r); nss.append(n_); oks.append(ok); errs.append(e)
        e_all = np.concatenate(errs)
        if v == 'base':
            base_err = e_all
            log(f'  SANITY base = {np.mean(rms):.2f} ± {np.std(rms):.2f} '
                f'(expected 7.56 ± 0.05)')
            assert abs(np.mean(rms) - 7.56) <= 0.02, 'base sanity FAILED'
            pv = '     —'
        else:
            p = st.wilcoxon(np.abs(e_all), np.abs(base_err),
                            zero_method='zsplit').pvalue
            pv = f'{p:.4f}'
        log(f'  {v:>8} | RUL {np.mean(rms):6.2f} ± {np.std(rms):5.2f} | '
            f'NASA {np.mean(nss):7.1f} | shape {int(np.sum(oks))}/3 | '
            f'p vs base {pv}')
        out[v] = dict(rms=np.array(rms), nss=np.array(nss),
                      oks=np.array(oks), err=e_all)
    np.savez(os.path.join(HERE, 'v5c_hi_ablate.npz'),
             **{f'{v}_{k}': val for v, d in out.items()
                for k, val in d.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
