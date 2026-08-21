"""
exp_hi_nc.py — STAGE 4: HI re-calibration on the NEW chain (kNN-cleaned model,
no gate).  Re-derives on dev, under the new data, the choices that were
previously selected under the old chain:
  * per-flight residual summary : mean_ng / median_ng / trim_ng  (ungated)
  * HI end level                : end_target in {1.00, 1.03}
  * Stage-5 beta                : nested dev-LOO NASA over BETA_C (rule frozen)
Metric: dev LOO truncation RUL RMSE / NASA (9 units x 4 fracs x 3 seeds),
shape criteria reported (start<=0.30, end>=0.93, tail monotone).
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import (HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok)
from exp_gate_ablation_nc import load_hi, SEEDS

VARIANTS = ['mean_ng', 'median_ng', 'trim_ng']
TARGETS = [1.00, 1.03]
RES = os.path.join(HERE, 'hi_nc_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('STAGE 4 — HI re-calibration, NEW chain (ungated summaries)')
    log('  variants x end_target, dev LOO truncation RUL; beta by nested LOO NASA')
    best = None
    for variant in VARIANTS:
        for tgt in TARGETS:
            cfg = dict(HI_CFG); cfg['end_target'] = tgt
            rmses, nasas, betas, shapes = [], [], [], []
            for sd in SEEDS:
                units, Z, hrs = load_hi(sd, variant)
                np.random.seed(sd)
                him, _ = train_model_tail([Z[u] for u in units], **cfg)
                HIs = {u: him.forward(Z[u]).flatten() for u in units}
                ok, (s0, e0, mono) = shape_ok(HIs)
                P, T, F, bsel, _ = loo('cycle', units, HIs, hrs, BETA_C)
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                nasas.append(nasa(P, T)); betas.append(bsel); shapes.append(ok)
            r = np.array(rmses)
            log(f'  {variant:>10} tgt={tgt:.2f}: RMSE {r.mean():.2f} ± {r.std():.2f}  '
                f'NASA {np.mean(nasas):.0f}  beta {betas}  '
                f'shape {sum(shapes)}/3')
            key = (variant, tgt)
            score = (0 if sum(shapes) == 3 else 1, r.mean())   # shape first
            if best is None or score < best[0]:
                best = (score, key, r, np.mean(nasas), betas)
    _, (bv, bt), br, bn, bb = best
    log('')
    log(f'SELECTED: {bv}, end_target={bt}  LOO RMSE={br.mean():.2f} ± {br.std():.2f}  '
        f'NASA={bn:.0f}  beta={bb}')
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
