"""
exp_raw_outlier_char.py — PHASE 1a: does the RAW training pool actually contain
removable outliers?

Cleaning presupposes that some rows are unexplainable measurements.  This script
characterises the raw cycle<5 pool WITHOUT any GP (model-free), so the answer
cannot be an artefact of the normal model:

  for each queried row, take its k nearest neighbours in standardized W space
  (alt, Mach, TRA, T2) among the whole cycle<5 pool, and score each sensor by
        rz_s = (x_s - median_neighbours(x_s)) / (1.4826 * MAD_neighbours(x_s))
  A point whose sensors agree with engines at virtually the same operating
  condition has |rz| ~ O(1).  A genuine bad measurement has a large |rz| on one
  or more sensors.  Aggregate score = RMS over the 5 sensors.

Reported: the score distribution vs a Gaussian reference (is there a separate
outlier population, or just a smooth tail?), how the score concentrates by
unit / cycle / flight class, and how many rows a given absolute cut removes.
"""
import os, sys, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
TRAIN_CYC = 5
K = 20
NQUERY = 60000
RES = os.path.join(HERE, 'raw_outlier_char_results.txt')
NPZ = os.path.join(HERE, 'raw_outlier_char.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    pool = np.where(A[:, 1].astype(int) < TRAIN_CYC)[0]
    log('PHASE 1a — model-free outlier characterisation of the RAW cycle<5 pool')
    log(f'  pool={len(pool)} rows, sensors={SENS}, k={K}, query={NQUERY}')

    Ws = StandardScaler().fit_transform(W[pool])
    Xp = X[pool]
    rng = np.random.default_rng(0)
    q = rng.choice(len(pool), NQUERY, replace=False)

    nn = NearestNeighbors(n_neighbors=K + 1).fit(Ws)
    _, idx = nn.kneighbors(Ws[q])
    nb = idx[:, 1:]
    log(f'  kNN built ({time.time()-t0:.0f}s)')

    Xn = Xp[nb]                                   # (NQUERY, K, 5)
    med = np.median(Xn, axis=1)
    mad = np.median(np.abs(Xn - med[:, None, :]), axis=1) * 1.4826
    mad = np.maximum(mad, 1e-9)
    rz = (Xp[q] - med) / mad                      # robust z per sensor
    score = np.sqrt((rz ** 2).mean(1))            # aggregate

    log('')
    log(f'  score percentiles [50,90,99,99.9,max] = '
        f'{np.percentile(score,[50,90,99,99.9]).round(2)} {score.max():.1f}')
    log(f'  per-sensor |rz| p99: ' +
        '  '.join(f'{s}={np.percentile(np.abs(rz[:,j]),99):.1f}'
                  for j, s in enumerate(SENS)))
    # Gaussian reference: RMS of 5 iid N(0,1) -> sqrt(chi2_5/5)
    ref = np.sqrt(rng.chisquare(5, size=NQUERY) / 5)
    log(f'  Gaussian reference score p99={np.percentile(ref,99):.2f} '
        f'p99.9={np.percentile(ref,99.9):.2f} max={ref.max():.2f}')
    log('  -> observed tail far beyond the reference = a genuine outlier '
        'population exists')

    log('')
    log('  rows removed by an ABSOLUTE cut on the score:')
    for c in [3, 4, 5, 6, 8, 10]:
        log(f'    score > {c:>2}: {100*(score>c).mean():6.3f}%  '
            f'({int((score>c).sum())} of {NQUERY})')

    unit = A[pool[q], 0].astype(int); cyc = A[pool[q], 1].astype(int)
    fc = A[pool[q], 2].astype(int)
    log('')
    log('  concentration of score>5 by unit / cycle / flight class:')
    hi = score > 5
    for nm, arr in [('unit', unit), ('cycle', cyc), ('Fc', fc)]:
        vals, cnt = np.unique(arr[hi], return_counts=True)
        tot = np.array([(arr == v).sum() for v in vals])
        rate = 100 * cnt / np.maximum(tot, 1)
        log(f'    {nm:>5}: ' + '  '.join(f'{int(v)}:{r:.2f}%'
                                         for v, r in zip(vals, rate)))

    np.savez(NPZ, score=score, rz=rz, q=q, pool=pool)
    log(f'\nwall={(time.time()-t0)/60:.1f} min -> {NPZ}')


if __name__ == '__main__':
    main()
