"""
exp_cleaning_absV.py — TRAINING-DATA cleaning on the ABSOLUTE detcov axis.

Replaces the percentile-labelled cleaning sweep (old Table 5, x = removed %)
with the colleague-code framing the user asked for:
      x = absolute detcov threshold V   (predictive-confidence index)
      y = healthy-generalization zRMSE
Selection criterion: V* = the threshold that MINIMIZES healthy-gen zRMSE.

Why absolute V: a percentile ("remove lowest d%") is dataset-specific and not
transferable; the absolute detcov value is the physical axis and lands on the
SAME scale as the inspection gate V*, unifying detcov's two uses.

Design per seed (frozen final config, identical to the deployed pipeline):
  sensors = T30,T48,T50,Nc,Wf ; RBF, coregionalization rank 1 ; Vecchia m=18 ;
  Adam lr 0.1 x 120 steps ; StandardScaler in/out.
  1. candidates = 16384 pts from cycle<5 rows of all dev units
  2. stage-1 GP on a plain 8192 subsample -> detcov of every candidate
  3. for each ABSOLUTE threshold V in a fixed grid (colleague style: keep
     detcov >= V, exactly as `np.where(detcov > 3.60)` in the DEMO notebook),
     resample 8192 from the kept set, retrain the GP.  V is the independent
     variable; the removed fraction floats and is only recorded.
  4. evaluate on two FIXED common sets (no test-time filtering):
       eval-gen : 8192 healthy pts (cycle>=5, pre-onset)  -> PRIMARY metric
       eval-in  : 8192 cycle<5 pts held out of the pool   -> coverage guard
     zRMSE uses per-sensor sigma frozen from the drop=0 baseline residuals.
Stores absolute V per (seed,f) so the figure is drawn on the absolute axis and
V* is read off directly.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
# reuse the exact frozen-config GP helpers; override sensors/rank via env below
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import (fit_gp, predict, detcov_of, true_onset_mask,
                                  SIDX, NPTS, NCAND, NEVAL, SEEDS, TRAIN_CYC)

# ABSOLUTE detcov threshold grid (colleague style: keep detcov >= V, hardcoded
# absolute value, NO fraction).  V is the independent variable; the removed
# fraction is recorded only as a derived by-product.  None = no-filter baseline.
VGRID = [None, 17.0, 19.0, 19.5, 19.7, 19.85, 20.0, 20.1, 20.2]
RES = os.path.join(HERE, 'cleaning_absV_results.txt')
NPZ = os.path.join(HERE, 'cleaning_absV_sweep.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    log('CLEANING SWEEP ON ABSOLUTE detcov AXIS  (final 5-sensor RBF rank1)')
    log(f'  V grid (absolute, keep detcov>=V)={VGRID}  seeds={SEEDS}')
    log('  V is the independent variable; removed fraction is a derived by-product')
    log('  select: min healthy-gen zRMSE -> V*')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy_gen = np.where(true_onset_mask(A))[0]
    log(f'  pool cycle<{TRAIN_CYC}: {len(pool)} | healthy-gen rows: {len(healthy_gen)}')

    R = {}       # (seed, V) -> dict
    frackept = {}  # (seed, V) -> removed fraction (derived)
    t00 = time.time()
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        perm = rng.permutation(len(pool))
        cand = pool[perm[:NCAND]]
        evin = pool[perm[NCAND:NCAND + NEVAL]]
        evgn = rng.choice(healthy_gen, NEVAL, replace=False)
        Yin, Ygn = X[evin], X[evgn]

        base_tr = cand[rng.choice(NCAND, NPTS, replace=False)]
        t0 = time.time()
        gp0 = fit_gp(W[base_tr], X[base_tr])
        dc_cand = detcov_of(gp0, W[cand])
        pcts = np.percentile(dc_cand, [1, 2, 5, 10, 50, 95])
        log(f'seed{sd}: stage-1 GP + candidate detcov ({time.time()-t0:.0f}s) '
            f'detcov[min,p1,p2,p5,p50,max]='
            f'[{dc_cand.min():.3f},{pcts[0]:.3f},{pcts[1]:.3f},{pcts[2]:.3f},'
            f'{pcts[4]:.3f},{dc_cand.max():.3f}]')

        zin0 = zgn0 = None
        for V in VGRID:
            t0 = time.time()
            if V is None:                                   # no-filter baseline
                gp = gp0; keep = cand; removed = 0.0
            else:
                keep = cand[dc_cand >= V]                   # ABSOLUTE threshold
                removed = 1.0 - len(keep) / NCAND
                repl = len(keep) < NPTS                     # guard: too-high V
                rng_d = np.random.default_rng(sd * 100 + int(round(V * 100)))
                tr = keep[rng_d.choice(len(keep), NPTS, replace=repl)]
                gp = fit_gp(W[tr], X[tr])
            frackept[(sd, V)] = removed
            r_in = Yin - predict(gp, W[evin])
            r_gn = Ygn - predict(gp, W[evgn])
            if V is None:
                zin0 = r_in.std(0); zgn0 = r_gn.std(0)
            z_in = float(np.sqrt(((r_in / zin0) ** 2).mean()))
            z_gn = float(np.sqrt(((r_gn / zgn0) ** 2).mean()))
            R[(sd, V)] = dict(z_in=z_in, z_gn=z_gn,
                              t48_gn=float(np.sqrt((r_gn[:, 1] ** 2).mean())))
            vlab = 'none' if V is None else f'{V:6.2f}'
            log(f'  seed{sd} V={vlab}  removed={removed:>5.1%}  '
                f'healthy-gen zRMSE={z_gn:.4f} (held-in {z_in:.4f})  '
                f'({time.time()-t0:.0f}s)')

    # ---- summary on the absolute V axis ----
    log('')
    log(f'{"V (abs)":>8} | {"removed% (mean)":>15} | {"healthy-gen zRMSE":>20} | '
        f'{"held-in zRMSE":>16} | {"T48 gen":>7}')
    log('-' * 82)
    gV, gZ, gZsd, gIn, gRem = [], [], [], [], []
    for V in VGRID:
        rem = np.mean([frackept[(sd, V)] for sd in SEEDS])
        zg = [R[(sd, V)]['z_gn'] for sd in SEEDS]
        zi = [R[(sd, V)]['z_in'] for sd in SEEDS]
        t48 = np.mean([R[(sd, V)]['t48_gn'] for sd in SEEDS])
        Vnum = np.nan if V is None else V
        gV.append(Vnum); gRem.append(rem)
        gZ.append(np.mean(zg)); gZsd.append(np.std(zg)); gIn.append(np.mean(zi))
        vlab = 'none' if V is None else f'{V:6.2f}'
        log(f'{vlab:>8} | {rem:>15.1%} | {np.mean(zg):>10.4f} ± {np.std(zg):<7.4f} | '
            f'{np.mean(zi):>16.4f} | {t48:>7.3f}')

    # V* = absolute threshold minimizing healthy-gen zRMSE (exclude no-filter)
    cand_i = [i for i, V in enumerate(VGRID) if V is not None]
    ibest = min(cand_i, key=lambda i: gZ[i])
    log('')
    log(f'V* (min healthy-gen zRMSE, absolute) = {gV[ibest]:.2f}  '
        f'removed={gRem[ibest]:.1%}  zRMSE={gZ[ibest]:.4f} ± {gZsd[ibest]:.4f}')
    np.savez(NPZ, V_grid=np.array(gV), removed_mean=np.array(gRem),
             zgn_mean=np.array(gZ), zgn_sd=np.array(gZsd), zin_mean=np.array(gIn),
             zgn_per_seed=np.array([[R[(sd, V)]['z_gn'] for V in VGRID] for sd in SEEDS]))
    log(f'\nwall total = {(time.time()-t00)/60:.1f} min  -> {NPZ}')


if __name__ == '__main__':
    main()
