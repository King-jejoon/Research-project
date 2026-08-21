"""
exp_v4_rul.py — v4 STAGE 7: official dev-side RUL numbers for the frozen v4
chain (set5 rbf r1, cycle<=3, gate V*=20.2, HI lambda1=8 lambda2=0.5).

Per seed: rebuild HI (frozen recipe), dev leave-one-unit-out truncation RUL on
the cycle axis, beta from the nested-LOO NASA rule (BETA_C grid).
Report: overall RMSE (mean +/- seed sd), per-truncation (20/40/60/80%) RMSE,
NASA score, beta per seed.  DS03 test stays SEALED in this script.
Outputs: v4_rul_results.txt, v4_rul.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, BETA_C, FRACS, nasa, loo, train_model_tail, \
    shape_ok
from exp_v4_hi import build_tables, SEEDS, VSTAR

L1, L2 = 8.0, 0.5                 # Stage 6 selection
RES = os.path.join(HERE, 'v4_rul_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'V4 STAGE 7 — dev LOO truncation RUL (cycle axis, beta by nested-LOO '
        f'NASA)')
    log(f'  chain: set5 rbf r1, cycle<=3 27k, gate V*={VSTAR} (absolute), '
        f'HI l1={L1:g} l2={L2:g}, frozen recipe')
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)

    rmses, nasas, betas = [], [], []
    frac_err = {f: [] for f in FRACS}
    for sd in SEEDS:
        units, Zn, hrs = build_tables(sd, VSTAR)
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: him.forward(Zn[u]).flatten() for u in units}
        ok, (st_, en_, mo_) = shape_ok(HIs)
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rmse = float(np.sqrt(((P - T) ** 2).mean()))
        ns = nasa(P, T)
        rmses.append(rmse); nasas.append(ns); betas.append(b)
        for f in FRACS:
            m = F == f
            frac_err[f].append(float(np.sqrt(((P[m] - T[m]) ** 2).mean())))
        log(f'  seed{sd}: shape={"OK" if ok else "FAIL"} (start {st_:.2f} '
            f'end {en_:.2f})  RMSE={rmse:.2f}  NASA={ns:.1f}  beta={b}  '
            f'({time.time()-t0:.0f}s)')

    log('')
    log(f'OFFICIAL v4 dev numbers (3 seeds):')
    log(f'  LOO RUL RMSE = {np.mean(rmses):.2f} ± {np.std(rmses):.2f}')
    log(f'  per truncation (20/40/60/80% of life): '
        + ' / '.join(f'{np.mean(frac_err[f]):.2f}' for f in FRACS))
    log(f'  NASA score = {np.mean(nasas):.1f} ± {np.std(nasas):.1f}')
    log(f'  beta per seed = {betas}')
    log(f'  [DS03 test remains SEALED — opening requires an explicit user '
        f'decision and is counted in the protocol]')
    np.savez(os.path.join(HERE, 'v4_rul.npz'),
             rmse=np.array(rmses), nasa=np.array(nasas),
             beta=np.array(betas),
             **{f'frac{int(100*f)}': np.array(frac_err[f]) for f in FRACS})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
