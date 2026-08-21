"""
exp_newclean_downstream.py — PHASE 1b: carry the validated raw-data cleaning
through the frozen chain and check the detector.

Phase 1 result: deleting rows whose model-free robust neighbour score exceeds 5
(3.74 % of the cycle<5 pool) lowers the median standardized residual by 6.1 %
and the trimmed zRMSE by 9.4 %, beating the matched random-removal control in
9 of 9 paired comparisons (sign test p = 0.004).

Here the final GP is rebuilt with that cleaning INSTEAD of the stage-1 detcov
5 % cleaning, and per-point statistics are produced for the 9 dev units exactly
as exp_sensor_study.py did, so the two can be compared like for like:
  * same model            : 5 sensors, RBF, rank 1, m=18, lr 0.1 x 120
  * same candidate draw   : rng 2000+seed, 16384 candidates
  * same final draw       : rng 3000+seed, 8192 training points
  * same per-unit sampling: rng unit*7+seed, 200 points per cycle
  * only the cleaning step differs
The cached sensor5_stats_s{seed}.npz IS the detcov-cleaned counterpart, so the
detector comparison needs no extra run.

Bonus: the new criterion needs no stage-1 GP at all (it never touches the
model), so building the final model is ~350 s per seed cheaper.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, SIDX, NPTS, NCAND, TRAIN_CYC, MCOND, DT
from demo_cond2 import conditional_stats2

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SEEDS = [0, 1, 2]
NPER = 200
CUT = float(os.environ.get('CUT', 5.0))
RES = os.path.join(HERE, 'newclean_downstream_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


@torch.no_grad()
def stats_of(gp, Wq, Xq, chunk=4096):
    dcs, lls = [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        _, dc, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                              gp['tY'], gp['struct'], teX,
                                              m=MCOND, test_y=teY)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(dcs), np.concatenate(lls)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'PHASE 1b — final model rebuilt with raw-data cleaning (score > {CUT} removed)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit_a = A[:, 0].astype(int); cyc_a = A[:, 1].astype(int); hs_a = A[:, 3]
    pool = np.where(cyc_a < TRAIN_CYC)[0]

    z = np.load(os.path.join(HERE, 'raw_clean.npz'))
    assert np.array_equal(z['pool'], pool), 'pool mismatch with cached scores'
    score = z['score']
    log(f'  pool={len(pool)}  removed by score>{CUT}: '
        f'{100*(score>CUT).mean():.2f}%  (no stage-1 GP needed)')

    for sd in SEEDS:
        out_path = os.path.join(HERE, f'newclean_stats_s{sd}.npz')
        if os.path.exists(out_path):
            log(f'seed{sd}: cached, skipping'); continue
        t0 = time.time()
        rng = np.random.default_rng(2000 + sd)          # same draw as sensor study
        cand = pool[rng.permutation(len(pool))[:NCAND]]
        _ = cand[rng.choice(NCAND, NPTS, replace=False)]  # keep rng stream aligned
        keep = cand[score[np.searchsorted(pool, cand)] <= CUT]
        rng2 = np.random.default_rng(3000 + sd)
        tr = keep[rng2.choice(len(keep), NPTS, replace=False)]
        gp = fit_gp(W[tr], X[tr])
        log(f'seed{sd}: final model built from {len(keep)} clean candidates '
            f'({time.time()-t0:.0f}s)  noise={gp["noise"]:.5f}')

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
            dcv, ll = stats_of(gp, W[idx], X[idx])
            hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
            below = np.where(hs_by < 0.5)[0]
            onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
            out[f'u{u}_cc'] = cc.astype(np.int32)
            out[f'u{u}_ll'] = ll.astype(np.float32)
            out[f'u{u}_dc'] = dcv.astype(np.float32)
            out[f'u{u}_onset'] = np.array([onset])
        np.savez(out_path, **out)
        log(f'seed{sd}: unit stats saved ({time.time()-t0:.0f}s total)')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
