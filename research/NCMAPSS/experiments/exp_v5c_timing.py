"""
exp_v5c_timing.py — unified running-time measurement of the residual
stage under the CURRENT 27k configuration (user go 2026-08-15).

Train = one fit on the 27,000-row training set, seed 0: cycle<=3 rows
for the stage-1 MOGP and the competitors (train_idx_27k), the recorded
v5c healthy-range rows for the stage-2 Proposed MOGP.  Fast models:
1 warm-up + median of 5 repeats; GP fits: a single timed run each.
Infer = residuals for the 6 test units on the chain evaluation grid
(NPER = 200, rng u*7+0): median of 3 repeats per model.
Machine state (NTHREADS, load average) is logged next to the numbers.
Outputs: v5c_timing_results.txt, v5c_timing.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
import exp_bench2_lib as B2
from exp_v4_c3_detcov import fit as gp_fit, point_stats, SIDX
from exp_v4_bench import train_idx_27k

SD = 0
NPER = 200
WARM, REPS, IREPS = 1, 5, 3
HP = {'lr': dict(), 'bspline': dict(n_knots=30), 'llke': dict(h=0.1),
      'cabn': dict(lam1=0.0)}
MAKERS = {'lr': B2.LinearNM, 'bspline': B2.BSplineNM,
          'llke': B2.LLKENM, 'cabn': B2.CaBNNM}
RES = os.path.join(HERE, 'v5c_timing_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'V5C TIMING — current 27k configuration, seed 0; '
        f'NTHREADS={torch.get_num_threads()}, '
        f'loadavg={tuple(round(v, 2) for v in os.getloadavg())}')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= 3)[0]
    tr1 = train_idx_27k(SD, unit, cyc, pool)
    tr2 = np.load(os.path.join(HERE, 'v5c_stats_s0.npz'))['train_idx']
    log(f'train rows: stage-1/competitors {len(tr1)}, stage-2 {len(tr2)}')

    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    ut = At[:, 0].astype(int); ct = At[:, 1].astype(int)
    idx = []
    for u in np.unique(ut):
        rows = np.where(ut == u)[0]; cyc_u = ct[rows]
        rng_u = np.random.default_rng(int(u) * 7 + SD)
        for c in np.unique(cyc_u):
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r)
    idx = np.concatenate(idx)
    log(f'inference grid: {len(idx)} points, 6 test units\n')

    out = {}
    for name, mk in MAKERS.items():
        ts = []
        m = None
        for i in range(WARM + REPS):
            m = mk(**HP[name])
            np.random.seed(0)
            t0 = time.time(); m.fit(W[tr1], X[tr1])
            if i >= WARM:
                ts.append(time.time() - t0)
        tis = []
        for _ in range(IREPS):
            t0 = time.time(); m.stats(Wt[idx], Xt[idx], chunk=4096)
            tis.append(time.time() - t0)
        tfit, tinf = float(np.median(ts)), float(np.median(tis))
        log(f'  {name:>8}: train {tfit:9.2f} s '
            f'(IQR {np.percentile(ts, 25):.2f}-{np.percentile(ts, 75):.2f})'
            f'  infer {tinf:8.2f} s '
            f'(IQR {np.percentile(tis, 25):.2f}-{np.percentile(tis, 75):.2f})')
        out[name] = (tfit, tinf)

    for tag, trr in [('mogp', tr1), ('proposed', tr2)]:
        t0 = time.time(); gp, nret = gp_fit(W[trr], X[trr])
        tfit = time.time() - t0
        log(f'  {tag:>8}: train {tfit:9.1f} s (single run, retries={nret})')
        tis = []
        for _ in range(IREPS):
            t0 = time.time(); point_stats(gp, Wt[idx], Xt[idx])
            tis.append(time.time() - t0)
        tinf = float(np.median(tis))
        log(f'  {tag:>8}: infer {tinf:8.2f} s '
            f'(IQR {np.percentile(tis, 25):.2f}-{np.percentile(tis, 75):.2f})')
        out[tag] = (tfit, tinf)

    np.savez(os.path.join(HERE, 'v5c_timing.npz'),
             **{f'{k}_train': np.float64(v[0]) for k, v in out.items()},
             **{f'{k}_infer': np.float64(v[1]) for k, v in out.items()})
    log(f'\nloadavg end={tuple(round(v, 2) for v in os.getloadavg())}  '
        f'wall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
