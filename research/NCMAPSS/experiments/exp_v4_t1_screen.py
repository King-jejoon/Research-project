"""
exp_v4_t1_screen.py — v4 STAGE 1: model-free sensor sensitivity screening
under the cycle<=3 training window (recomputation of Table 1; the old T1
reused cycle<5 outputs).

Method (identical to the documented T1 definition):
  * reference       : kNN k=10 healthy reference over STANDARDIZED operating
                      conditions W=[alt, Mach, TRA, T2]; reference pool =
                      dev rows with cycle <= 3 (the new training window)
  * healthy noise   : per-sensor sigma = kNN-residual s.d. on HEALTHY rows
                      outside the window (hs >= 0.5, cycle > 3) — this bakes
                      the operating-condition coverage error of the window
                      reference into sigma, so an insensitive sensor scores
                      z ~= 1 (same convention as the old T1, whose bottom
                      ranks sit at <= 1.02)
  * score           : zRMS of degraded residuals (hs < 0.5 rows) in
                      healthy-noise units; ranking is scale-free
  * subsampling     : ref 150k / healthy held-out 30k / degraded 60k (rng 0)
Output: ranking of all 14 sensors, top-3/5/7 candidate sets,
        v4_t1_screen.npz, v4_t1_results.txt
"""
import os, sys, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L

TRAIN_CYC_MAX = 3
NREF, NHOLD, NDEG = 150_000, 30_000, 60_000
K = 10
RES = os.path.join(HERE, 'v4_t1_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'V4 T1 — model-free sensor screening, window cycle<={TRAIN_CYC_MAX}, '
        f'kNN k={K}')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    cyc = A[:, 1].astype(int); hs = A[:, 3]
    rng = np.random.default_rng(0)

    pool = np.where(cyc <= TRAIN_CYC_MAX)[0]
    deg = np.where(hs < 0.5)[0]
    healthy_out = np.where((hs >= 0.5) & (cyc > TRAIN_CYC_MAX))[0]
    log(f'  pool cycle<={TRAIN_CYC_MAX}: {len(pool)}  '
        f'healthy outside window: {len(healthy_out)}  '
        f'degraded (hs<0.5): {len(deg)}')

    ref = rng.choice(pool, min(NREF, len(pool)), replace=False)
    hold = rng.choice(healthy_out, min(NHOLD, len(healthy_out)), replace=False)
    dsm = rng.choice(deg, min(NDEG, len(deg)), replace=False)

    ws = StandardScaler().fit(W[ref])
    nn = NearestNeighbors(n_neighbors=K, n_jobs=-1).fit(ws.transform(W[ref]))

    def resid(idx):
        _, nb = nn.kneighbors(ws.transform(W[idx]))
        return X[idx] - X[ref][nb].mean(axis=1)

    r_hold = resid(hold)
    sig = r_hold.std(0) + 1e-12
    r_deg = resid(dsm)
    z = np.sqrt(((r_deg / sig) ** 2).mean(0))

    order = np.argsort(-z)
    log('')
    log(f'  {"rank":>4} | {"sensor":>6} | zRMS (healthy-noise units)')
    for i, j in enumerate(order):
        log(f'  {i+1:>4} | {L.OUTPUT_NAMES[j]:>6} | {z[j]:.3f}')
    sets = {n: [L.OUTPUT_NAMES[j] for j in order[:n]] for n in (3, 5, 7)}
    log('')
    for n in (3, 5, 7):
        log(f'  top-{n}: {sets[n]}')
    log('  [old T1 (cycle<5 outputs): T50 T48 Wf Nc T30 P40 Ps30 T24 ...; '
        'old sets 3=[T48,T50,Wf] 5=[T30,T48,T50,Nc,Wf] '
        '7=[T24,T30,T48,T50,Ps30,Nc,Wf]]')
    np.savez(os.path.join(HERE, 'v4_t1_screen.npz'),
             zrms=z, order=order,
             names=np.array(L.OUTPUT_NAMES),
             sig=sig)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
