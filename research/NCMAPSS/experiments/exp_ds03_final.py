"""
exp_ds03_final.py — THE official DS03-012 pipeline, end to end, one command.
Confirmed operating configuration (2026-07-17):

  Stage 1  GP normal model : dev cycle<5, 8192 pts, 3 sensors (T48,T50,Wf),
                             RBF rank 2, m=18, Adam lr 0.1 x 120 steps
                             (kernel & lr selected by GP-level held-out fit)
  Stage 2  conditional stats: all dev+test units, NPER=200 samples/cycle
                             (bug-fixed whitening, demo_cond2)  [cached npz]
  Stage 3a detection (monitoring only): EWMA(lambda=.2, K=3) on cycle-mean m2
  Stage 3b aggregation      : plain cycle-mean residuals -> dev z-normalisation
                             (detcov filter REMOVED 2026-07-17 — no accuracy
                              contribution under the exponential Stage 5)
  Stage 4  HI network      : MLP 3-4-2-1, lambda=(init 1, end 1, mono 8,
                             conv 2) + L_flat(w=800, m=0.004), 1000 ep
  (2026-07-17 revision: Stage-5 basis exponential — start-is-minimum constraint)
  Stage 5  RUL             : Bayesian EXPONENTIAL regression g0+g2(e^bc-1),
                             beta by dev leave-one-unit-out CV; h=1 first passage
  Eval     truncation 20/40/60/80% x 6 test units x 3 seeds, RMSE + NASA score

Official result this must reproduce: RMSE 7.23 +/- 0.10, NASA score 17.8.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_ds03_final.py
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_ds03_port as EP
from exp_ds03_port import nasa_score, FRACS
from stage5_exp import rul_eval_exp
from exp_gp_selfdrop import build_resid, detect_acc
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
DROP = 0.0
SENS = ['T48', 'T50', 'Wf']
GP_CFG = dict(kernel='rbf', rank=2, m=18, npts=8192, steps=120, mcond=15, lr=0.1)
HI_CFG = dict(epochs=1000, lambda0=1.0, lambda1=8.0, lambda2=2.0,
              init_threshold=0.2, flat_w=800.0, flat_m=0.004, alpha=0.001)
TAG = 's3_rbf_r2_m18_n8192_p200_lr0.1'
RES = os.path.join(HERE, 'final_pipeline_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('DS03-012 OFFICIAL PIPELINE  (confirmed 2026-07-17)')
    log(f'  GP {GP_CFG} | NPER=200 | no filter (plain cycle mean) | HI {HI_CFG}')
    EP.NPER = 200
    cache = L.load_cache()
    sidx = [L.OUTPUT_NAMES.index(s) for s in SENS]
    cfg = dict(sens=SENS, sidx=sidx, tag=TAG, **GP_CFG)

    allP, allT, allFR = [], [], []
    zs = {}
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        path = os.path.join(HERE, f'port_stats_{TAG}_s{sd}.npz')
        zs[sd] = EP.compute_stats(cache, sd, cfg, path)       # Stages 1-2 (cached)
        dl, tl = build_resid(zs[sd], DROP)                    # Stage 3b
        np.random.seed(sd)
        him, _ = train_model_tail([d for _, _, d in dl], **HI_CFG)   # Stage 4
        dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
        tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
        P, T, FR, UN, beta = rul_eval_exp(dHI, tHI, FRACS)    # Stage 5 (exp basis, dev-CV beta)
        log(f"  seed{sd}: dev-CV beta = {beta}")
        allP.append(P); allT.append(T); allFR.append(FR)
        rmse = float(np.sqrt(((P - T) ** 2).mean()))
        log(f'  seed{sd}: RUL RMSE {rmse:.2f}  score {nasa_score(P, T):.1f}  '
            f'({time.time()-t0:.0f}s)')

    a, dly = np.mean([detect_acc(zs[sd]) for sd in SEEDS], 0)  # Stage 3a
    log(f'\nDETECTION (monitoring): acc {a:.1f}%  |delay| {dly:.1f} cyc')
    log(f'{"truncation":>10} | {"RMSE mean":>9} | {"score mean":>10}')
    for f in FRACS:
        rs = [float(np.sqrt(((P[FR == f] - T[FR == f]) ** 2).mean()))
              for P, T, FR in zip(allP, allT, allFR)]
        scs = [nasa_score(P[FR == f], T[FR == f]) for P, T, FR in zip(allP, allT, allFR)]
        log(f'{f:>9.0%} | {np.mean(rs):>9.2f} | {np.mean(scs):>10.1f}')
    rmses = np.array([float(np.sqrt(((P - T) ** 2).mean())) for P, T in zip(allP, allT)])
    scores = np.array([nasa_score(P, T) for P, T in zip(allP, allT)])
    log(f'\nOFFICIAL: RUL RMSE = {rmses.mean():.2f} +/- {rmses.std():.2f}   '
        f'NASA score = {scores.mean():.1f} +/- {scores.std():.1f}')
    log(f'(expected: 7.23 +/- 0.10 / 17.8)   wall {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
