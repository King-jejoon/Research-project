"""
exp_kernel_cv.py — STEP 1: kernel x rank cross-validation for the Vecchia MOGP.

Grid : {rbf, matern52, matern32, matern12} x rank {1,2,3} x seeds {0,1,2}
Data : DS03 dev, cycle<5 pool; per seed ONE fixed 8192-pt training set and ONE
       fixed 8192-pt held-out set shared by ALL 12 combos (pure model comparison).
Inputs = [alt, Mach, TRA, T2] (colleague's alt-drop bug fixed), outputs = T48/T50/Wf.
Metric: held-out actual-vs-predicted RMSE — per-sensor raw units + combined in
        train-scaler z-space (same scaler for all combos within a seed -> fair).
Frozen everything else: m=18, 120 steps, lr=0.1, ARD, StandardScaler.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2

DT = torch.float32
SENS = os.environ.get('SENSORS', 'T48,T50,Wf').split(',')
TAG = os.environ.get('TAG', '')
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
KERNELS = ['rbf', 'matern52', 'matern32', 'matern12']
RANKS = [1, 2, 3]
SEEDS = [0, 1, 2]
NPTS, NEVAL = 8192, 8192
M, STEPS, LR, MCOND = 18, 120, 0.1, 15
TRAIN_CYC = 5
RES = os.path.join(HERE, f'kernel_cv{TAG}_results.txt')
NPZ = os.path.join(HERE, f'kernel_cv{TAG}.npz')


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
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


@torch.no_grad()
def predict(gp, Wq, chunk=4096):
    out = []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        pred_s, *_ = conditional_stats2(gp['model'], gp['lik'], gp['tX'], gp['tY'],
                                        gp['struct'], teX, m=MCOND, test_y=None)
        out.append(pred_s * gp['ys'].scale_ + gp['ys'].mean_)
    return np.concatenate(out)


def main():
    open(RES, 'w').close()
    log(f'STEP1 KERNEL x RANK CV  kernels={KERNELS} ranks={RANKS} seeds={SEEDS}')
    log(f'  n_train={NPTS} n_eval={NEVAL} (cycle<{TRAIN_CYC}, shared per seed)  '
        f'sensors={SENS}  inputs=[alt,Mach,TRA,T2]  m={M} steps={STEPS} lr={LR}')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    pool = np.where(A[:, 1].astype(int) < TRAIN_CYC)[0]

    Z = {}   # (kernel, rank) -> list over seeds of zRMSE
    R = {}   # per-sensor raw rmse
    t00 = time.time()
    for sd in SEEDS:
        rng = np.random.default_rng(1000 + sd)
        perm = rng.permutation(len(pool))
        tr = pool[perm[:NPTS]]
        ev = pool[perm[NPTS:NPTS + NEVAL]]
        Yev = X[ev]
        yscale = StandardScaler().fit(X[tr]).scale_
        for kernel in KERNELS:
            for rank in RANKS:
                t0 = time.time()
                try:
                    gp = fit(W[tr], X[tr], kernel, rank)
                    pred = predict(gp, W[ev])
                    resid = Yev - pred
                    rmse = np.sqrt((resid ** 2).mean(0))
                    z = float(np.sqrt(((resid / yscale) ** 2).mean()))
                except Exception as e:
                    log(f'  seed{sd} {kernel:>9} r{rank}: FAILED {type(e).__name__}: '
                        f'{str(e)[:100]}')
                    rmse = np.full(len(SENS), np.nan); z = np.nan
                Z.setdefault((kernel, rank), []).append(z)
                R.setdefault((kernel, rank), []).append(rmse)
                log(f'  seed{sd} {kernel:>9} r{rank}: zRMSE={z:.4f}  '
                    + '  '.join(f'{s}={v:.3f}' for s, v in zip(SENS, rmse))
                    + f'  ({time.time()-t0:.0f}s)')

    log('')
    log(f'{"kernel":>9} {"rank":>4} | {"zRMSE mean±sd":>16} | {"T48":>7} {"T50":>7} '
        f'{"Wf":>8} | rank-order')
    log('-' * 72)
    order = sorted(Z, key=lambda k: np.nanmean(Z[k]))
    npz = {}
    for i, key in enumerate(order):
        zs = np.array(Z[key]); rr = np.nanmean(np.stack(R[key]), 0)
        log(f'{key[0]:>9} r{key[1]:>3} | {np.nanmean(zs):>8.4f} ± {np.nanstd(zs):<6.4f}'
            f' | {rr[0]:>7.3f} {rr[1]:>7.3f} {rr[2]:>8.4f} | #{i+1}')
        npz[f'{key[0]}_r{key[1]}'] = zs
    best = order[0]
    log(f'\nWINNER: kernel={best[0]}  rank={best[1]}  '
        f'zRMSE={np.nanmean(Z[best]):.4f}±{np.nanstd(Z[best]):.4f}')
    n2 = order[1]
    log(f'runner-up: {n2[0]} r{n2[1]}  zRMSE={np.nanmean(Z[n2]):.4f} '
        f'(gap {100*(np.nanmean(Z[n2])/np.nanmean(Z[best])-1):.1f}%)')
    np.savez(NPZ, **npz)
    log(f'wall = {(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
