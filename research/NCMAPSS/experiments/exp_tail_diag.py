"""
exp_tail_diag.py — PHASE 0 DIAGNOSIS: what makes the residual tail?

The frozen GP's healthy residuals have a very heavy tail (top 1% of points
carry ~1/3 of the total squared error).  Raw-data cleaning can only help if
that tail is BAD DATA.  If instead the tail is systematic physics (thermal lag
during throttle/altitude transients: the same operating point W maps to
different sensor values depending on history), then no amount of training-set
cleaning will move the error, and the right lever is a feature / a transient
gate.  This script decides which.

Frozen model (unchanged): 5 sensors T30,T48,T50,Nc,Wf; RBF; coregionalization
rank 1; Vecchia m=18; Adam lr 0.1 x 120; StandardScaler.  Trained on 8192 pts
from cycle<5 of the dev units — the exact stage-1 GP of the cleaning sweep.

Evaluated on the SAME fixed healthy-generalization set (8192 pts, cycle>=5,
pre-onset) that the cleaning sweeps use, reproduced from the same rng stream.

Tests
  T1 spatial repeatability : for each eval point, |z| vs the mean |z| of its
     k=10 nearest neighbours in standardized W space (self excluded), against
     a label-permutation null.  High correlation => the SAME operating point is
     consistently mispredicted => systematic (physics/model), not noise.
  T2 transient association : |z| binned by |dTRA/dt|, |dalt/dt|, |dT2/dt|
     (per-second differences along the time-ordered rows of each flight).
  T3 flight-phase position : |z| by relative position within the flight.
  T4 sensor breakdown      : which of the 5 sensors carries the tail.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import (fit_gp, predict, true_onset_mask, SENS, SIDX,
                                  NPTS, NCAND, NEVAL, TRAIN_CYC)

SEED = int(os.environ.get('SEED', 0))
RES = os.path.join(HERE, 'tail_diag_results.txt')
NPZ = os.path.join(HERE, 'tail_diag.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def rate_features(W, A):
    """per-row |d/dt| of alt, Mach, TRA, T2 along time-ordered rows, with the
    first row of every (unit,cycle) segment masked to the segment's second row."""
    d = np.abs(np.diff(W, axis=0, prepend=W[:1]))
    same = (A[:, 0] == np.roll(A[:, 0], 1)) & (A[:, 1] == np.roll(A[:, 1], 1))
    same[0] = False
    d[~same] = 0.0
    return d          # columns: alt, Mach, TRA, T2 rates


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('PHASE 0 — residual-tail diagnosis (frozen 5-sensor RBF rank1 GP)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy_gen = np.where(true_onset_mask(A))[0]

    # ---- reproduce the cleaning sweep's fixed sets from the same rng stream ----
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(pool))
    cand = pool[perm[:NCAND]]
    _evin = pool[perm[NCAND:NCAND + NEVAL]]
    evgn = rng.choice(healthy_gen, NEVAL, replace=False)
    base_tr = cand[rng.choice(NCAND, NPTS, replace=False)]

    t0 = time.time()
    gp = fit_gp(W[base_tr], X[base_tr])
    log(f'  stage-1 GP fitted ({time.time()-t0:.0f}s)  noise={gp["noise"]:.5f}')
    pred = predict(gp, W[evgn])
    r = X[evgn] - pred
    sig = r.std(0)
    z = r / sig                       # per-sensor standardized residual
    zn = np.sqrt((z ** 2).mean(1))    # per-point magnitude across sensors
    log(f'  eval pts={len(zn)}  |z| median={np.median(zn):.3f} '
        f'p95={np.percentile(zn,95):.3f} p99={np.percentile(zn,99):.3f} max={zn.max():.1f}')
    top1 = np.sort(zn ** 2)[-len(zn) // 100:].sum() / (zn ** 2).sum()
    log(f'  top 1% of points carry {100*top1:.0f}% of total squared error')

    # ---- T1 spatial repeatability ------------------------------------------
    from sklearn.neighbors import NearestNeighbors
    Ws = gp['xs'].transform(W[evgn])
    nn = NearestNeighbors(n_neighbors=11).fit(Ws)
    _, idx = nn.kneighbors(Ws)
    nb = idx[:, 1:]                                   # exclude self
    nbmean = zn[nb].mean(1)
    c_obs = float(np.corrcoef(zn, nbmean)[0, 1])
    rs = np.random.default_rng(0)
    null = [float(np.corrcoef(zp := rs.permutation(zn), zp[nb].mean(1))[0, 1])
            for _ in range(5)]
    log('')
    log('T1 spatial repeatability (k=10 neighbours in standardized W):')
    log(f'   corr(|z|, neighbour mean |z|) = {c_obs:.4f}   '
        f'permutation null = {np.mean(null):+.4f} +- {np.std(null):.4f}')
    log('   -> high = the SAME operating point is consistently mispredicted '
        '(systematic); ~0 = i.i.d. measurement noise')

    # ---- T2 transient association ------------------------------------------
    D = rate_features(W, A)[evgn]
    names = ['|d alt/dt|', '|d Mach/dt|', '|d TRA/dt|', '|d T2/dt|']
    log('')
    log('T2 transient association (eval points binned by rate, 5 equal bins):')
    log(f'   {"feature":>12} | ' + ' | '.join(f'Q{i+1}' for i in range(5)) + ' | ratio Q5/Q1')
    t2 = {}
    for j, nm in enumerate(names):
        e = np.percentile(D[:, j], [0, 20, 40, 60, 80, 100])
        mm = []
        for i in range(5):
            m = (D[:, j] >= e[i]) & (D[:, j] <= e[i + 1] if i == 4 else D[:, j] < e[i + 1])
            mm.append(zn[m].mean() if m.any() else np.nan)
        t2[nm] = mm
        log(f'   {nm:>12} | ' + ' | '.join(f'{v:4.2f}' for v in mm) +
            f' | {mm[4]/mm[0]:6.2f}x')

    # ---- T3 position within flight -----------------------------------------
    key = A[evgn, 0].astype(np.int64) * 10000 + A[evgn, 1].astype(np.int64)
    pos = np.zeros(len(evgn))
    for k in np.unique(key):
        m = key == k
        seg = np.where((A[:, 0].astype(np.int64) * 10000 +
                        A[:, 1].astype(np.int64)) == k)[0]
        pos[m] = (evgn[m] - seg[0]) / max(1, len(seg) - 1)
    log('')
    log('T3 position within flight (0=start, 1=end), mean |z| in 5 bins:')
    pb = [zn[(pos >= i / 5) & (pos < (i + 1) / 5 if i < 4 else pos <= 1)].mean()
          for i in range(5)]
    log('   ' + ' | '.join(f'{v:4.2f}' for v in pb))

    # ---- T4 sensor breakdown ------------------------------------------------
    log('')
    log('T4 per-sensor share of total squared standardized error:')
    sh = (z ** 2).sum(0) / (z ** 2).sum()
    for s, v in zip(SENS, sh):
        log(f'   {s:>5}: {100*v:5.1f}%')

    np.savez(NPZ, zn=zn, z=z, rates=D, pos=pos, corr_obs=c_obs,
             corr_null=np.array(null), sensors=np.array(SENS))
    log(f'\nwall={(time.time()-t00)/60:.1f} min  -> {NPZ}')


if __name__ == '__main__':
    main()
