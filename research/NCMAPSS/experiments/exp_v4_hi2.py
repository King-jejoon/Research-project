"""
exp_v4_hi2.py — v4 STAGE 6 (revision): extended HI lambda grid after the
u11 diagnosis.

Changes vs exp_v4_hi.py, both justified on dev only:
  * grid extended: lambda1 in {2,4,6,8,12,16} x lambda2 in {0.25,0.5,1,2,4}
  * selection composite drops Distribution shift (user decision) ->
    3 axes: Monotonicity, Curvature, Information range
Everything else identical (gate V*=20.2 absolute, frozen loss terms, shape
gate start<=0.30 / end>=0.93 / monotone tail, 3 seeds; LOO RUL as sanity
column only).
Outputs: v4_hi2_results.txt, v4_hi2.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, loo, train_model_tail, shape_ok
from exp_v4_hi import build_tables, properties, SEEDS, VSTAR

L1S = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]
L2S = [0.25, 0.5, 1.0, 2.0, 4.0]
RES = os.path.join(HERE, 'v4_hi2_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'V4 STAGE 6r — extended HI grid (V*={VSTAR} abs, trim25, '
        f'composite without Shift)')
    data = {sd: build_tables(sd, VSTAR) for sd in SEEDS}

    R = {}
    for l1 in L1S:
        for l2 in L2S:
            cfg = dict(HI_CFG); cfg.update(lambda1=l1, lambda2=l2,
                                           end_target=1.03)
            props, rmses, oks = [], [], []
            for sd in SEEDS:
                units, Zn, hrs = data[sd]
                np.random.seed(sd)
                him, _ = train_model_tail([Zn[u] for u in units], **cfg)
                HIs = {u: him.forward(Zn[u]).flatten() for u in units}
                ok, _ = shape_ok(HIs)
                P, T, F, bsel, _ = loo('cycle', units, HIs, hrs, BETA_C)
                props.append(properties(HIs))
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                oks.append(ok)
            pr = np.array(props).mean(0)
            R[(l1, l2)] = dict(mon=pr[0], curv=pr[1], rng=pr[2], shift=pr[3],
                               rul=float(np.mean(rmses)),
                               rul_sd=float(np.std(rmses)),
                               shape=int(np.sum(oks)))
            log(f'  l1={l1:>4} l2={l2:>4}: Mon={pr[0]:8.2f} Curv={pr[1]:6.2f} '
                f'Range={pr[2]:5.3f} Shift={pr[3]:6.4f} '
                f'LOO RUL={np.mean(rmses):.2f}±{np.std(rmses):.2f} '
                f'shape {int(np.sum(oks))}/3')

    ok_keys = [k for k in R if R[k]['shape'] == 3]
    pool_keys = ok_keys if ok_keys else list(R)
    arr = {p: np.array([R[k][p] for k in pool_keys])
           for p in ['mon', 'curv', 'rng']}

    def norm(v, sign):
        v = sign * v
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    comp = (norm(arr['mon'], +1) + norm(arr['curv'], +1)
            + norm(arr['rng'], +1)) / 3
    order = np.argsort(-comp)
    log('')
    log(f'shape-passing combos ({len(ok_keys)}): {sorted(ok_keys)}')
    log('composite ranking (Mon+Curv+Range, no Shift), top 6:')
    for i in order[:6]:
        k = pool_keys[i]
        log(f'  {k}: comp={comp[i]:.3f}  LOO RUL={R[k]["rul"]:.2f}')
    sel = pool_keys[int(order[0])]
    log(f'SELECTED: lambda1={sel[0]}, lambda2={sel[1]}  '
        f'(LOO RUL {R[sel]["rul"]:.2f}±{R[sel]["rul_sd"]:.2f}; grid best '
        f'{min(R.values(), key=lambda r: r["rul"])["rul"]:.2f})')
    np.savez(os.path.join(HERE, 'v4_hi2.npz'),
             l1=np.array([sel[0]]), l2=np.array([sel[1]]),
             vstar=np.array([VSTAR]),
             **{f'{k[0]}_{k[1]}_{p}': R[k][p] for k in R
                for p in ['mon', 'curv', 'rng', 'shift', 'rul']},
             **{f'{k[0]}_{k[1]}_shape': R[k]['shape'] for k in R})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
