"""
exp_ds01_final.py — transfer the CONFIRMED DS03-012 pipeline, unchanged, to
N-CMAPSS DS02-006 (colleague original dataset, HPT+LPT).

Everything identical to exp_ds03_final.py: GP cycle<5 8192pts s3/m32r1/lr.05,
NPER=200 conditional stats, per-cycle causal detcov drop 85%, HI with
lambda=(1,1,2,0.25)+L_flat(300,.002), Bayesian first-passage RUL,
truncation 20/40/60/80% x 3 seeds. Only the dataset changes.

Builds cache_ds02.npz on first run (only the 6 arrays the pipeline needs).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_ds01_final.py
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

H5 = os.path.join(__import__('exp_paths').DATA, 'N-CMAPSS_DS02-006.h5')
CACHE_DS01 = os.path.join(HERE, 'cache_ds02.npz')
SEEDS = [0, 1, 2]
DROP = 0.85
SENS = ['T48', 'T50', 'Wf']
GP_CFG = dict(kernel='matern32', rank=1, m=18, npts=8192, steps=120, mcond=15, lr=0.05)
HI_CFG = dict(epochs=1000, lambda0=1.0, lambda1=2.0, lambda2=0.25,
              init_threshold=0.2, flat_w=300.0, flat_m=0.002, alpha=0.001)
TAG = 'ds02_s3_matern32_r1_m18_n8192_p200'
RES = os.path.join(HERE, 'final_pipeline_ds02_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def load_ds01():
    if not os.path.exists(CACHE_DS01):
        import h5py
        log('[cache] building cache_ds02.npz from h5 ...')
        with h5py.File(H5, 'r') as h:
            arrs = {k: np.array(h[k], dtype=np.float32)
                    for k in ['W_dev', 'X_s_dev', 'A_dev', 'W_test', 'X_s_test', 'A_test']}
        np.savez_compressed(CACHE_DS01, **arrs)
        log(f'[cache] saved ({os.path.getsize(CACHE_DS01)/1e6:.0f} MB)')
    return dict(np.load(CACHE_DS01, allow_pickle=False))


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('DS02-006 TRANSFER — final pipeline (exponential Stage 5), unchanged')
    log(f'  GP {GP_CFG} | NPER=200 | causal drop {DROP:.0%} | HI {HI_CFG}')
    EP.NPER = 200
    cache = load_ds01()
    for split in ['dev', 'test']:
        A = cache[f'A_{split}']
        units = np.unique(A[:, 0].astype(int))
        lens = [int(A[A[:, 0] == u, 1].max()) for u in units]
        log(f'  {split}: units {list(units)}  lifetimes {lens}')
    sidx = [L.OUTPUT_NAMES.index(s) for s in SENS]
    cfg = dict(sens=SENS, sidx=sidx, tag=TAG, **GP_CFG)

    allP, allT, allFR = [], [], []
    zs = {}
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        path = os.path.join(HERE, f'port_stats_{TAG}_s{sd}.npz')
        zs[sd] = EP.compute_stats(cache, sd, cfg, path)
        log(f'  seed{sd}: stats ready ({time.time()-t0:.0f}s)')
        dl, tl = build_resid(zs[sd], DROP)
        np.random.seed(sd)
        him, _ = train_model_tail([d for _, _, d in dl], **HI_CFG)
        dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
        tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
        FR5 = [round(f, 2) for f in np.arange(0.05, 0.96, 0.05)]
        P, T, FR, UN, beta = rul_eval_exp(dHI, tHI, FR5)
        log(f"  seed{sd}: dev-CV beta = {beta}")
        allP.append(P); allT.append(T); allFR.append(FR)
        m4 = np.isin(FR, FRACS)
        rmse = float(np.sqrt(((P[m4] - T[m4]) ** 2).mean()))
        log(f'  seed{sd}: RUL RMSE {rmse:.2f}  score {nasa_score(P[m4], T[m4]):.1f}  '
            f'({time.time()-t0:.0f}s)')

    a, dly = np.mean([detect_acc(zs[sd]) for sd in SEEDS], 0)
    log(f'\nDETECTION (monitoring): acc {a:.1f}%  |delay| {dly:.1f} cyc')
    log(f'{"truncation":>10} | {"RMSE mean":>9} | {"score mean":>10}')
    for f in [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]:
        rs = [float(np.sqrt(((P[FR == f] - T[FR == f]) ** 2).mean()))
              for P, T, FR in zip(allP, allT, allFR)]
        scs = [nasa_score(P[FR == f], T[FR == f]) for P, T, FR in zip(allP, allT, allFR)]
        log(f'{f:>9.0%} | {np.mean(rs):>9.2f} | {np.mean(scs):>10.1f}')
    rmses = np.array([float(np.sqrt(((P[np.isin(FR, FRACS)] - T[np.isin(FR, FRACS)]) ** 2).mean()))
                      for P, T, FR in zip(allP, allT, allFR)])
    scores = np.array([nasa_score(P[np.isin(FR, FRACS)], T[np.isin(FR, FRACS)])
                       for P, T, FR in zip(allP, allT, allFR)])
    log(f'\nDS02-006: RUL RMSE = {rmses.mean():.2f} +/- {rmses.std():.2f}   '
        f'NASA score = {scores.mean():.1f} +/- {scores.std():.1f}')
    log(f'(exp-S5 references: DS03 7.84 +/- 0.12 / 19.9, DS01 5.99 +/- 0.18 / 10.4)   wall {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
