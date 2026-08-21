"""
exp_step5_threshold.py — STEP 4+5: final model, then detcov inspection
threshold calibrated on the ABSOLUTE detcov axis via onset-detection RMSE.

STEP 4 (per seed): candidates 16384 (cycle<5) -> stage-1 GP (rbf r1)
  -> candidate detcov -> drop lowest 5% -> resample 8192 -> FINAL GP.
STEP 5:
  A. per dev unit (9), every cycle, 200 pts/cycle: per-point LL (fixed
     whitening) + detcov under the FINAL model.  Cached to npz per seed.
  B. threshold sweep on absolute detcov values V (grid = quantile positions
     of the pooled dev detcov, reported as absolute values):
       keep detcov >= V -> per-cycle mean LL (cycle w/o kept pts: unfiltered
       fallback) -> mean-drop changepoint -> detected onset
     metric: onset RMSE (detected - true hs onset) over 9 units, per seed;
     also mean delay, |delay|<=3 hit rate, kept fraction.
  Selection: V* = argmin onset RMSE (consistency checked across seeds).
  V* transfers to test units AS AN ABSOLUTE VALUE (no re-percentiling).
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from demo_cond2 import conditional_stats2

os.environ.setdefault('KERNEL', 'rbf')
os.environ.setdefault('RANK', '1')
from exp_traindata_detcov import fit_gp, detcov_of, SENS, SIDX, DT

NPTS, NCAND = 8192, 16384
CLEAN = 0.05
SEEDS = [0, 1, 2]
NPER, MCOND, TRAIN_CYC = 200, 15, 5
QPOS = [0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30, 40]   # quantile positions -> abs V
RES = os.path.join(HERE, 'step5_threshold_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def ewma_first_crossing(curve, warmup=10, lam=0.2, kk=3.0):
    """First cycle index whose EWMA of LL crosses below the healthy LCL.

    mean-drop changepoint fails here: degradation is accelerating, so the
    best mean-split lands at the unit end.  Onset = FIRST departure from the
    healthy baseline (first `warmup` cycles), EWMA-smoothed.
    """
    x = np.asarray(curve, float)
    ic = x[:warmup]
    mu0, sd0 = ic.mean(), ic.std(ddof=1) + 1e-8
    lcl = mu0 - kk * sd0 * np.sqrt(lam / (2 - lam))
    e = mu0
    for i, v in enumerate(x):
        e = lam * v + (1 - lam) * e
        if i >= warmup and e < lcl:
            return i
    return len(x)


def main():
    open(RES, 'w').close()
    log(f'STEP4+5  final model = rbf r1 + {CLEAN:.0%} train-clean;  '
        f'threshold sweep on ABSOLUTE detcov axis (grid at quantile positions {QPOS})')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit_a = A[:, 0].astype(int); cyc_a = A[:, 1].astype(int); hs_a = A[:, 3]
    pool = np.where(cyc_a < TRAIN_CYC)[0]
    t00 = time.time()

    sweep = {q: [] for q in QPOS + [0]}   # qpos -> list over seeds of per-unit delays
    absV = {}
    for sd in SEEDS:
        stats_path = os.path.join(HERE, f'step5_stats_s{sd}.npz')
        if not os.path.exists(stats_path):
            # ---- STEP 4: final model ----
            t0 = time.time()
            rng = np.random.default_rng(2000 + sd)
            cand = pool[rng.permutation(len(pool))[:NCAND]]
            tr1 = cand[rng.choice(NCAND, NPTS, replace=False)]
            gp1 = fit_gp(W[tr1], X[tr1])
            dc = detcov_of(gp1, W[cand])
            thr5 = np.percentile(dc, CLEAN * 100)
            keep = cand[dc >= thr5]
            rng2 = np.random.default_rng(3000 + sd)
            tr = keep[rng2.choice(len(keep), NPTS, replace=False)]
            gp = fit_gp(W[tr], X[tr])
            log(f'seed{sd}: final model built ({time.time()-t0:.0f}s)')

            # ---- STEP 5A: per-unit stats under final model ----
            out = {}
            for u in np.unique(unit_a):
                rows = np.where(unit_a == u)[0]
                cyc_u = cyc_a[rows]; hs_u = hs_a[rows]
                rng_u = np.random.default_rng(int(u) * 7 + sd)
                ucyc = np.unique(cyc_u)
                idx, cc = [], []
                for c in ucyc:
                    r = rows[cyc_u == c]
                    if len(r) > NPER:
                        r = rng_u.choice(r, NPER, replace=False)
                    idx.append(r); cc.append(np.full(len(r), c))
                idx = np.concatenate(idx); cc = np.concatenate(cc)
                ll = np.empty(len(idx)); dcv = np.empty(len(idx))
                for i in range(0, len(idx), 4096):
                    sl = idx[i:i + 4096]
                    teX = torch.tensor(gp['xs'].transform(W[sl]), dtype=DT)
                    teY = torch.tensor(gp['ys'].transform(X[sl]), dtype=DT)
                    _, dci, _, llf, _ = conditional_stats2(
                        gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'],
                        teX, m=MCOND, test_y=teY)
                    ll[i:i + 4096] = llf; dcv[i:i + 4096] = dci
                hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
                below = np.where(hs_by < 0.5)[0]
                onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
                out[f'u{u}_cc'] = cc.astype(np.int32)
                out[f'u{u}_ll'] = ll.astype(np.float32)
                out[f'u{u}_dc'] = dcv.astype(np.float32)
                out[f'u{u}_onset'] = np.array([onset])
            np.savez_compressed(stats_path, **out)
            log(f'seed{sd}: unit stats cached ({time.time()-t0:.0f}s total)')
        z = np.load(stats_path)
        units = sorted({int(k[1:].split('_')[0]) for k in z.files})
        dc_pool = np.concatenate([z[f'u{u}_dc'] for u in units])
        Vgrid = {q: float(np.percentile(dc_pool, q)) for q in QPOS}
        absV[sd] = Vgrid

        # ---- STEP 5B: sweep ----
        for q in [0] + QPOS:
            V = -np.inf if q == 0 else Vgrid[q]
            delays = []
            for u in units:
                cc, ll, dc = z[f'u{u}_cc'], z[f'u{u}_ll'], z[f'u{u}_dc']
                onset = int(z[f'u{u}_onset'][0])
                ucyc = np.unique(cc)
                curve = []
                for c in ucyc:
                    mk = (cc == c) & (dc >= V)
                    if not mk.any():
                        mk = cc == c
                    curve.append(ll[mk].mean())
                k = ewma_first_crossing(np.array(curve))
                det = int(ucyc[k]) if k < len(ucyc) else int(ucyc[-1] + 1)
                delays.append(det - onset)
            sweep[q].append(delays)
        log(f'seed{sd}: sweep done  ({(time.time()-t00)/60:.1f} min cum)')

    # ---- report ----
    log('')
    log(f'{"q-pos":>6} {"abs V (s0/s1/s2)":>24} | {"kept%":>6} | {"onset RMSE":>10} | '
        f'{"mean delay":>10} | {"|d|<=3":>7}')
    log('-' * 84)
    best = None
    for q in [0] + QPOS:
        D = np.array(sweep[q])                     # seeds x units
        rmse = float(np.sqrt((D.astype(float) ** 2).mean()))
        md = float(D.mean()); hit = float((np.abs(D) <= 3).mean())
        vs = 'no filter' if q == 0 else '/'.join(f'{absV[sd][q]:.3f}' for sd in SEEDS)
        log(f'{q:>6} {vs:>24} | {100-q:>5.1f} | {rmse:>10.2f} | {md:>+10.2f} | '
            f'{100*hit:>6.1f}%')
        if best is None or rmse < best[1]:
            best = (q, rmse)
    log(f'\nBEST: quantile position {best[0]}%  onset RMSE={best[1]:.2f} cycles')
    if best[0] != 0:
        log('  absolute V* per seed: ' +
            ', '.join(f'seed{sd}: {absV[sd][best[0]]:.4f}' for sd in SEEDS))
    log('  (V* transfers to test units as the absolute value, no re-percentiling)')
    log(f'wall = {(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
