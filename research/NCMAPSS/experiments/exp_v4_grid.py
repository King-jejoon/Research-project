"""
exp_v4_grid.py — v4 STAGE 2 (full grid): sensor set x kernel x rank under the
cycle<=3 window (27k train).  One process per sensor set (SET env: 3|5|7).

Candidate sets = top-3/5/7 of the v4 T1 screening (exp_v4_t1_screen.py,
window cycle<=3):
  3=[T48,T50,Wf]  5=[T30,T48,T50,Nc,Wf]  7=[T30,T48,T50,Ps30,P40,Nc,Wf]
(set7 follows the ranking honestly: P40 in, T24 out — the old set7 deviated
from its own T1 ranking on that seat.)

Grid per set: kernels {rbf, matern52, matern32, matern12} x ranks {1,2,3}
              x 3 seeds = 36 fits.
Data (per seed, identical for every config within the seed, shared row indices
across sets): cycle<=3 rows, stratified 1000 per (unit,cycle) -> 27,000 train;
8,192 held-out eval rows.  Vecchia m=18, Adam lr 0.1 x 120, StandardScaler.
Metric: held-out zRMSE in scaled output space (comparable within a set).
Cholesky failures recorded; failed seeds excluded from the config mean.
Per-set selection: min mean zRMSE; below-seed-sd differences -> simpler
(RBF first, lowest rank).  Cross-set comparison happens downstream (onset
RMSE, Stage 3).
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2

DT = torch.float32
SETS = {3: ['T48', 'T50', 'Wf'],
        5: ['T30', 'T48', 'T50', 'Nc', 'Wf'],
        7: ['T30', 'T48', 'T50', 'Ps30', 'P40', 'Nc', 'Wf']}
SET = int(os.environ.get('SET', 5))
SENS = SETS[SET]
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
TRAIN_CYC_MAX, NPER_TRAIN, NEVAL = 3, 1000, 8192
KERNELS = ['rbf', 'matern52', 'matern32', 'matern12']
SEEDS = [0, 1, 2]
M, STEPS, LR, MCOND = 18, 120, 0.1, 15
RES = os.path.join(HERE, f'v4_grid_set{SET}_results.txt')
NPZ = os.path.join(HERE, f'v4_grid_set{SET}.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit(Wtr, Ytr, kernel, rank):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT)
    tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=len(SENS), rank=rank,
                         kernel=kernel).to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=len(SENS)).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=M, num_steps=STEPS, lr=LR,
                           group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure,
                xs=xs, ys=ys)


@torch.no_grad()
def zrmse(gp, Wq, Xq, chunk=4096):
    """held-out RMSE in the model's scaled output space."""
    errs = []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        pred_s, *_ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                        gp['tY'], gp['struct'], teX,
                                        m=MCOND, test_y=None)
        errs.append(pred_s - gp['ys'].transform(Xq[i:i + chunk]))
    e = np.concatenate(errs)
    return float(np.sqrt((e ** 2).mean()))


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'V4 STAGE 2 GRID — set{SET} {SENS}  kernels x ranks {[1,2,3]}  '
        f'(cycle<={TRAIN_CYC_MAX}, 27k train)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= TRAIN_CYC_MAX)[0]

    splits = {}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        tr = []
        for u in np.unique(unit[pool]):
            for c in range(1, TRAIN_CYC_MAX + 1):
                rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
                tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)),
                                     replace=False))
        tr = np.concatenate(tr)
        rest = np.setdiff1d(pool, tr, assume_unique=False)
        ev = rng.choice(rest, NEVAL, replace=False)
        splits[sd] = (tr, ev)
    log(f'  splits fixed: train={len(splits[0][0])}, eval={NEVAL} per seed; '
        f'shared by every config within a seed')

    results = {}

    def run(kernel, rank):
        vals = []
        for sd in SEEDS:
            tr, ev = splits[sd]
            t0 = time.time()
            try:
                gp = fit(W[tr], X[tr], kernel, rank)
                v = zrmse(gp, W[ev], X[ev])
                vals.append(v)
                log(f'  {kernel:>9} r{rank} seed{sd}: zRMSE={v:.5f}  '
                    f'({time.time()-t0:.0f}s)')
            except torch._C._LinAlgError:
                vals.append(np.nan)
                log(f'  {kernel:>9} r{rank} seed{sd}: CHOLESKY FAIL  '
                    f'({time.time()-t0:.0f}s)')
        arr = np.array(vals)
        results[(kernel, rank)] = arr
        ok = arr[~np.isnan(arr)]
        log(f'  -> {kernel} r{rank}: mean={np.mean(ok):.5f} ± {np.std(ok):.5f}  '
            f'(fails {int(np.isnan(arr).sum())}/3)')
        return np.mean(ok) if len(ok) else np.inf

    log('')
    log('FULL GRID — kernels x ranks')
    means = {}
    for k in KERNELS:
        for r in [1, 2, 3]:
            means[(k, r)] = run(k, r)

    log('')
    log('summary (mean zRMSE, fails in parentheses):')
    log(f'  {"kernel":>9} | ' + ' | '.join(f'{"rank "+str(r):>16}'
                                           for r in [1, 2, 3]))
    for k in KERNELS:
        cells = []
        for r in [1, 2, 3]:
            arr = results[(k, r)]; nf = int(np.isnan(arr).sum())
            ok = arr[~np.isnan(arr)]
            cells.append(f'{np.mean(ok):.5f}±{np.std(ok):.5f}'
                         + (f'({nf}F)' if nf else '    '))
        log(f'  {k:>9} | ' + ' | '.join(f'{c:>16}' for c in cells))

    best = min(means, key=means.get)
    sd_best = np.nanstd(results[best])
    simple_order = {k: i for i, k in enumerate(KERNELS)}
    cands = [c for c in means if means[c] - means[best] <= sd_best]
    sel = sorted(cands, key=lambda c: (simple_order[c[0]], c[1]))[0]
    log('')
    log(f'best by mean: {best[0]} r{best[1]} ({means[best]:.5f}); '
        f'within-sd candidates: {sorted(cands)}')
    log(f'SELECTED (simplicity rule): kernel={sel[0]}, rank={sel[1]}')
    np.savez(NPZ, **{f'{k}_r{r}': v for (k, r), v in results.items()},
             winner_kernel=np.array([sel[0]]), winner_rank=np.array([sel[1]]))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
