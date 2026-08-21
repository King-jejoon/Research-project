"""
exp_dsx_hi1.py — DS01/DS02 transfer, PHASE 1c: HI lambda1 x lambda2 grid
re-selection on the stage-1 ungated cycle<=3 tables + HI #1 chain numbers.
Dev only, cached stats, no test contact.

Protocol byte-identical to exp_v5c_lambda_grid / exp_v4_hi2: same grid
{2,4,6,8,12,16} x {0.25,0.5,1,2,4}, same frozen loss-term structure
(lambda0/thr/end_target/flat w/m stay recipe constants — user decision
2026-08-19), same composite (min-max normalized Mon + Curv + Range, no
Shift), same shape gate, med3-free inside the grid.  Selection: shape
3/3 eligibility -> composite top (no tie-band predecessor here — this is
the dataset's first selection).  Then the HI #1 chain numbers are logged
with the selected recipe + med3 + beta by the frozen dev-LOO NASA rule.
Usage: DS=ds01|ds02 python3 exp_dsx_hi1.py
Outputs: {DS}v6_hi1_results.txt, {DS}v6_hi1.npz (selected recipe, chain dev)
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, train_model_tail, \
    shape_ok
from exp_v4_final import med3
from exp_dsx_lib import tables_ungated, run_grid
SEEDS = [int(x) for x in os.environ.get('SEEDS_ONLY', '0,1,2').split(',')]

DS = os.environ.get('DS', 'ds01')
assert DS in ('ds01', 'ds02', 'ds07', 'ds08a', 'ds04', 'ds05', 'ds06', 'ds08c')
RES = os.path.join(HERE, f'{DS}v6_hi1_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'{DS.upper()} V6 HI1 — lambda grid on stage-1 ungated cycle<=3 '
        'tables (DS03 protocol; first selection for this dataset)')
    data = {sd: (lambda t: (t[0], t[2], t[3]))(tables_ungated(DS, sd))
            for sd in SEEDS}
    R, sel, comp = run_grid(data, 'stage-1 grid', log)
    log(f'SELECTED: lambda1={sel[0]}, lambda2={sel[1]} '
        f'(comp {comp[sel]:.3f}, LOO RUL {R[sel]["rul"]:.2f}'
        f'±{R[sel]["rul_sd"]:.2f})')

    # HI #1 chain numbers with the selected recipe (med3, frozen beta rule)
    cfg = dict(HI_CFG); cfg.update(lambda1=sel[0], lambda2=sel[1],
                                   end_target=1.03)
    rms, nss, oks, bs = [], [], [], []
    for sd in SEEDS:
        units, Zn, hrs = data[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
        ok, _ = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean())))
        nss.append(nasa(P, T)); oks.append(int(ok)); bs.append(b)
        log(f'chain seed{sd}: dev LOO={rms[-1]:.2f} '
            f'shape={"OK" if ok else "FAIL"} beta={b}')
    log('')
    log(f'HI #1 chain (cycle<=3 ungated, recipe {sel}, med3): dev LOO = '
        f'{np.mean(rms):.2f} ± {np.std(rms):.2f}  '
        f'NASA {np.mean(nss):.1f} ± {np.std(nss):.1f}  '
        f'shape {int(np.sum(oks))}/{len(SEEDS)}  beta={bs}')
    np.savez(os.path.join(HERE, f'{DS}v6_hi1.npz'),
             l1=np.array([sel[0]]), l2=np.array([sel[1]]),
             dev_rm=np.array(rms), dev_ns=np.array(nss),
             beta=np.array(bs),
             **{f'{k[0]}_{k[1]}_{p}': R[k][p] for k in R
                for p in ['mon', 'curv', 'rng', 'shift', 'rul']})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
