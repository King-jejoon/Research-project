"""
exp_tail_diag2.py — PHASE 0b: is the clustered error a W-REGION effect or a
UNIT/FLIGHT effect?  (no GP needed: reuses tail_diag.npz residuals)

T1 found corr(|z|, neighbour-mean |z|) = 0.90 in standardized W space.  Two very
different readings:
  (i)  neighbours are mostly the SAME unit/flight  -> the offset is a per-unit or
       per-flight bias (a flight-level "hard to explain" case; cleanable at
       flight granularity)
  (ii) neighbours are mostly OTHER units           -> whole regions of the
       operating envelope are mispredicted for every engine (model/coverage
       deficiency; point removal cannot fix it)
Also reports how concentrated the error is by unit and by cycle.
"""
import os, sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
from exp_traindata_detcov import true_onset_mask, NPTS, NCAND, NEVAL, TRAIN_CYC

RES = os.path.join(HERE, 'tail_diag2_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    zn = np.load(os.path.join(HERE, 'tail_diag.npz'))['zn']
    cache = L.load_cache()
    W, A = cache['W_dev'], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy_gen = np.where(true_onset_mask(A))[0]

    # reproduce the exact rng stream of exp_tail_diag.py (seed 0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(pool))
    cand = pool[perm[:NCAND]]
    _evin = pool[perm[NCAND:NCAND + NEVAL]]
    evgn = rng.choice(healthy_gen, NEVAL, replace=False)
    base_tr = cand[rng.choice(NCAND, NPTS, replace=False)]
    assert len(evgn) == len(zn)

    xs = StandardScaler().fit(W[base_tr])          # same scaler as the GP
    Ws = xs.transform(W[evgn])
    unit = A[evgn, 0].astype(int); cycle = A[evgn, 1].astype(int)

    nn = NearestNeighbors(n_neighbors=11).fit(Ws)
    dist, idx = nn.kneighbors(Ws)
    nb, nbd = idx[:, 1:], dist[:, 1:]

    same_u = (unit[nb] == unit[:, None])
    same_f = same_u & (cycle[nb] == cycle[:, None])
    log('PHASE 0b — W-region effect vs unit/flight effect')
    log(f'  neighbour composition (k=10): same unit {100*same_u.mean():.1f}%  '
        f'same flight {100*same_f.mean():.1f}%')
    log(f'  neighbour distance in standardized W: median {np.median(nbd):.4f} '
        f'p95 {np.percentile(nbd,95):.4f}')

    full = np.corrcoef(zn, zn[nb].mean(1))[0, 1]
    cross_mean = np.full(len(zn), np.nan)
    for i in range(len(zn)):
        m = ~same_u[i]
        if m.any():
            cross_mean[i] = zn[nb[i][m]].mean()
    ok = ~np.isnan(cross_mean)
    cross = np.corrcoef(zn[ok], cross_mean[ok])[0, 1]
    log('')
    log(f'  corr(|z|, neighbour mean |z|)              = {full:.4f}   (all neighbours)')
    log(f'  corr(|z|, CROSS-UNIT neighbour mean |z|)   = {cross:.4f}   '
        f'({ok.sum()} of {len(zn)} points have a cross-unit neighbour)')
    log('  -> cross-unit correlation staying high = the operating REGION is bad '
        'for every engine (not a per-unit bias)')

    log('')
    log('  error concentration by unit:')
    for u in np.unique(unit):
        m = unit == u
        log(f'    unit {u:>2}: n={m.sum():>5}  mean|z|={zn[m].mean():.3f}  '
            f'share of total z^2 = {100*(zn[m]**2).sum()/(zn**2).sum():5.1f}%')

    q = np.percentile(zn, 99)
    top = zn >= q
    log('')
    log(f'  top-1% points (|z|>={q:.2f}): {top.sum()} pts, '
        f'{len(np.unique(unit[top]))} distinct units, '
        f'{len(np.unique(cycle[top]))} distinct cycles')
    log(f'    unit histogram: {dict(zip(*np.unique(unit[top], return_counts=True)))}')
    log(f'    their neighbours are same-unit {100*same_u[top].mean():.1f}% of the time')
    log(f'    mean |z| of their neighbours = {zn[nb[top]].mean():.3f} '
        f'(overall mean {zn.mean():.3f})')


if __name__ == '__main__':
    main()
