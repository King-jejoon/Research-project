"""
exp_v4_bench_resid.py — per-model residual construction on the full dev life.

Each benchmark model is trained exactly like the chain's GP (seed 0, the
27,000-row v4 training set, CV-winning hyperparameters), then residuals are
computed for every dev unit and cycle on the chain's own sampling grid
(200 rows per cycle, rng(u*7+seed)).  mogp reuses v4_c3_stats_s0.npz.
Per-sensor residuals are expressed in each model's own healthy-noise units
(std over the training-window cycles), so every panel has noise floor ~1 and
the degradation signal-to-noise is directly comparable across models.
Output: v4_bench_resid.npz, fig scratchpad/bench_resid_models.png
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_bench2_lib as B2
from exp_v4_bench import train_idx_27k

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
SD = 0
NPER = 200
HP = {'llke': dict(h=0.1), 'bspline': dict(n_knots=30),
      'lr': dict(), 'cabn': dict(lam1=0.0)}
MAKERS = {'lr': B2.LinearNM, 'bspline': B2.BSplineNM,
          'llke': B2.LLKENM, 'cabn': B2.CaBNNM}


def main():
    t0 = time.time()
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int); hs = A[:, 3]
    pool = np.where(cyc <= 3)[0]
    tr = train_idx_27k(SD, unit, cyc, pool)

    # chain sampling grid (identical to v4 stats)
    idx_all, cc_all, uu_all, onsets = [], [], [], {}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]; hs_u = hs[rows]
        rng_u = np.random.default_rng(int(u) * 7 + SD)
        ucyc = np.unique(cyc_u)
        for c in ucyc:
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx_all.append(r); cc_all.append(np.full(len(r), c))
            uu_all.append(np.full(len(r), u))
        hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
        below = np.where(hs_by < 0.5)[0]
        onsets[int(u)] = int(ucyc[below[0]]) if len(below) else int(ucyc[-1]+1)
    idx_all = np.concatenate(idx_all)
    cc_all = np.concatenate(cc_all).astype(int)
    uu_all = np.concatenate(uu_all).astype(int)

    out = {'cc': cc_all, 'uu': uu_all,
           'onsets': np.array([onsets[u] for u in sorted(onsets)]),
           'units': np.array(sorted(onsets))}

    # mogp residuals from the chain cache
    Z = np.load(os.path.join(HERE, f'v4_c3_stats_s{SD}.npz'))
    rs, cs, us = [], [], []
    for u in sorted(onsets):
        rs.append(Z[f'u{u}_resid'])
        cs.append(Z[f'u{u}_cc']); us.append(np.full(len(Z[f'u{u}_cc']), u))
    out['mogp_resid'] = np.concatenate(rs)
    out['mogp_cc'] = np.concatenate(cs).astype(int)
    out['mogp_uu'] = np.concatenate(us).astype(int)
    print(f'mogp: reused chain cache ({time.time()-t0:.0f}s)', flush=True)

    for name, maker in MAKERS.items():
        t1 = time.time()
        m = maker(**HP[name])
        np.random.seed(0)
        m.fit(W[tr], X[tr])
        pred, _, _ = m.stats(W[idx_all], X[idx_all], chunk=4096)
        out[f'{name}_resid'] = (X[idx_all] - pred).astype(np.float32)
        print(f'{name}: fitted + {len(idx_all)} residuals '
              f'({time.time()-t1:.0f}s)', flush=True)

    np.savez(os.path.join(HERE, 'v4_bench_resid.npz'), **out)
    print(f'saved. wall={(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
