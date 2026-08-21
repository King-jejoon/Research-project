"""
exp_rank2.py — Stage-1 coregionalization rank 1 -> 2 (s3, matern32), evaluated
under the FINAL system (per-cycle causal detcov filter + L_flat HI + RUL).

The old 2x2 grid only tested rank 2 with RBF (s3/rbf-r2 8.85, old HI).
matern32+rank2 is untested; this closes it against the rank-1 control
(ds_drop 0/60/85% = 8.60/8.20/8.05 under the identical downstream).

Stats cached to port_stats_s3_matern32_r2_m18_n8192_p200_s<sd>.npz.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_rank2.py
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_ds03_port as EP
from exp_gp_selfdrop import detect_acc, downstream, HI_KW, SEEDS, DS_DROPS, SENS

TAG = 's3_matern32_r2_m18_n8192_p200'
RES = os.path.join(HERE, 'rank2_results.txt')
CTRL = {0.0: (8.60, 0.03, 27.2), 0.60: (8.20, 0.09, 24.0), 0.85: (8.05, 0.11, 23.5)}


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def main():
    open(RES, 'a').close()
    log(f'\n===== RANK 2 coregionalization (s3, matern32)  {time.strftime("%m-%d %H:%M")} =====')
    log('  GP: 3 sensors, matern32, rank=2, m=18, npts=8192, steps=120, lr=0.05, '
        'NPER=200 | downstream: causal filter + final HI(L_flat) + RUL')
    EP.NPER = 200
    cache = L.load_cache()
    sidx = [L.OUTPUT_NAMES.index(s) for s in SENS]
    cfg = dict(sens=SENS, sidx=sidx, kernel='matern32', rank=2, m=18, npts=8192,
               steps=120, mcond=15, drop=0.0, detect='ewma', tag=TAG, lr=0.05)

    zs = {}
    for sd in SEEDS:
        t0 = time.time()
        torch.manual_seed(sd); np.random.seed(sd)
        path = os.path.join(HERE, f'port_stats_{TAG}_s{sd}.npz')
        zs[sd] = EP.compute_stats(cache, sd, cfg, path)
        log(f'  seed{sd}: stats ready ({time.time()-t0:.0f}s)')

    a, dly = np.mean([detect_acc(zs[sd]) for sd in SEEDS], 0)
    log(f'\n[rank2]  det acc={a:.1f}%  |delay|={dly:.1f}   (rank1 ref: ~82%)')
    for dsd, r, rs, sc, s9, s10, ok in downstream(zs, HI_KW):
        c = CTRL[dsd]
        log(f'  ds_drop={dsd:.0%}: RMSE {r:.2f} ±{rs:.2f}  score {sc:.1f}  '
            f's9->s10 {s9:.2f}->{s10:.2f}  {"PASS" if ok else "fail"}   '
            f'| rank1: {c[0]:.2f} ±{c[1]:.2f} / {c[2]:.1f}')
    log('done.')


if __name__ == '__main__':
    main()
