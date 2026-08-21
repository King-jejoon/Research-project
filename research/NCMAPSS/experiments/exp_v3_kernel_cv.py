"""
exp_v3_kernel_cv.py — STAGE A1 of the v3 chain: kernel x rank selection under
the NEW data configuration.

Data (per seed, identical for every configuration within the seed):
  train : cycle<3 rows, stratified 1000 per (unit, cycle) -> 18,000 points
  eval  : 8,192 held-out cycle<3 rows (never trained on)
Model family: 5 sensors T30/T48/T50/Nc/Wf, Vecchia m=18, Adam lr 0.1 x 120,
StandardScaler in/out, inputs W=[alt,Mach,TRA,T2].

Two phases in one run:
  P1  kernels {rbf, matern52, matern32, matern12} x rank 1 x 3 seeds
  P2  winner kernel x ranks {2, 3} x 3 seeds
Metric: held-out zRMSE in the model's scaled output space (comparable across
configs because train set — hence scaler — is shared within a seed).
Cholesky failures are recorded and the config's failed seed is excluded from
its mean (same policy as the old chain; instability is itself a result).
Selection: min mean zRMSE; differences below seed s.d. -> simpler config.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2

DT = torch.float32
SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
TRAIN_CYC, NPER_TRAIN, NEVAL = 3, 1000, 8192
KERNELS = ['rbf', 'matern52', 'matern32', 'matern12']
SEEDS = [0, 1, 2]
M, STEPS, LR, MCOND = 18, 120, 0.1, 15
RES = os.path.join(HERE, 'v3_kernel_cv_results.txt')
NPZ = os.path.join(HERE, 'v3_kernel_cv.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit(Wtr, Ytr, kernel, rank):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT)
    tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=len(SENS), rank=rank, kernel=kernel).to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(SENS)).to('cpu', DT)
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
    log('V3 STAGE A1 — kernel x rank CV  (cycle<3, 1000/(unit,cycle) = 18k train)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]

    splits = {}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        tr = []
        for u in np.unique(unit[pool]):
            for c in range(1, TRAIN_CYC):
                rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
                tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)), replace=False))
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
    log('P1 — kernels at rank 1')
    p1 = {k: run(k, 1) for k in KERNELS}
    order = sorted(p1, key=p1.get)
    win = order[0]
    # simplicity tie-break: prefer rbf if within one pooled seed s.d.
    sd_ref = np.nanstd(results[(win, 1)])
    if win != 'rbf' and p1['rbf'] - p1[win] <= sd_ref:
        log(f'  {win} best but rbf within seed s.d. ({p1["rbf"]:.5f} vs '
            f'{p1[win]:.5f}, sd {sd_ref:.5f}) -> rbf by simplicity rule')
        win = 'rbf'
    log(f'P1 winner: {win}')

    log('')
    log(f'P2 — {win} at ranks 2, 3')
    for r in [2, 3]:
        run(win, r)
    means = {r: np.nanmean(results[(win, r)]) for r in [1, 2, 3]}
    sds = {r: np.nanstd(results[(win, r)]) for r in [1, 2, 3]}
    best_r = min(means, key=means.get)
    final_r = 1 if means[1] - means[best_r] <= sds[best_r] else best_r
    log('')
    log(f'rank means: ' + '  '.join(f'r{r}={means[r]:.5f}±{sds[r]:.5f}'
                                    for r in [1, 2, 3]))
    log(f'SELECTED: kernel={win}, rank={final_r} '
        f'(simplicity rule applied: {final_r != best_r or best_r == 1})')
    np.savez(NPZ, **{f'{k}_r{r}': v for (k, r), v in results.items()},
             winner_kernel=np.array([win]), winner_rank=np.array([final_r]))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
