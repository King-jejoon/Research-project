"""
Scaling comparison on the 3 HI sensors {T48,T50,Wf} (the ones that feed the
neural Health Index), so it is fast AND directly relevant.

  previous ExactGP (capped ~180 pts)  vs  colleague MOGP-Vecchia (scales up)

Config (from M3 Max benchmarks): device=cpu, dtype=float32, torch threads=6.
Results appended to mogp_scale3_results.txt after EACH level.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp, vecchia_mogp_predictive_mean

DEV, DT = 'cpu', torch.float32
RANK, KERNEL, M, STEPS, LR = 2, 'rbf', 20, 150, 0.05
SENS = ['T48', 'T50', 'Wf']
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]     # [2,3,13]
PER_RANGE = [100, 300, 600, 1200]                  # ~900 / 2700 / 5400 / 10800
ABN_EVAL = 3000
RES = os.path.join(HERE, 'mogp_scale3_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def build_traj(resid, A, abn_idx, abn_cyc):
    units = A[abn_idx, 0].astype(int); cycles = abn_cyc.astype(int)
    out = {}
    for u in np.unique(units):
        um = units == u; out[int(u)] = {}
        uc = cycles[um]; ur = resid[um]; uq = np.unique(uc)
        for j, name in enumerate(SENS):
            out[int(u)][name] = (uq, np.array([ur[uc == cc, j].mean() for cc in uq]))
    return out


def metrics(mean_val, mean_abn, Yv, Ya, A, abn_idx, abn_cyc):
    rmse = float(np.sqrt(((Yv - mean_val) ** 2).mean()))
    hstd = {name: float((Yv[:, j] - mean_val[:, j]).std()) for j, name in enumerate(SENS)}
    traj = build_traj(Ya - mean_abn, A, abn_idx, abn_cyc)
    snr = L.degradation_snr(traj, hstd, SENS)
    return rmse, float(np.nanmean([snr[s] for s in SENS]))


def fit_predict_mogp(Wtr, Ytr, m, steps):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT); tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(Wtr.shape[1], num_tasks=Ytr.shape[1], rank=RANK, kernel=KERNEL).to(DEV, DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Ytr.shape[1]).to(DEV, DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=LR, group=True, verbose=False)
    model.eval(); lik.eval()

    def predict(Wte):
        teX = torch.tensor(xs.transform(Wte), dtype=DT)
        with torch.no_grad():
            mu = vecchia_mogp_predictive_mean(model, lik, tX, tY, teX, r.structure, m=m)
        return mu.cpu().numpy() * ys.scale_ + ys.mean_
    return predict


def main():
    open(RES, 'w').close()
    log(f'config: cpu float32 threads=6 rank={RANK} kernel={KERNEL} m={M} steps={STEPS} '
        f'| sensors={SENS} (HI inputs)')
    c = L.load_cache()
    W, Xall, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    X = Xall[:, SIDX]
    nr = c['normal_ranges']; dni = c['dev_normal_idx']
    dai, dac = c['dev_abn_indices'], c['dev_abn_cycles']
    if len(dai) > ABN_EVAL:
        sub = np.random.default_rng(0).choice(len(dai), ABN_EVAL, replace=False)
        dai, dac = dai[sub], dac[sub]

    pools = {pr: L.dense_healthy_pool(nr, per_range=pr, seed=1, exclude=dni) for pr in PER_RANGE}
    excl = np.unique(np.concatenate([dni] + list(pools.values())))
    val = L.dense_healthy_pool(nr, per_range=150, seed=99, exclude=excl)
    Yv, Ya = X[val], X[dai]
    log(f'val={len(val)} abn_eval={len(dai)}')
    log('=' * 60)
    log(f'{"method":>16} | {"n_train":>7} | {"RMSE":>7} | {"SNR":>6} | {"sec":>6}')
    log('-' * 60)

    # previous ExactGP on 3 sensors, 180 pts
    t = time.time()
    m0, l0, sc0 = L.train_gp_arrays(W[dni], X[dni], iters=100, seed=0)
    mv, _ = L.gp_predict(m0, l0, sc0, W, val); ma, _ = L.gp_predict(m0, l0, sc0, W, dai)
    rmse, snr = metrics(mv, ma, Yv, Ya, A, dai, dac)
    log(f'{"ExactGP (prev)":>16} | {len(dni):7d} | {rmse:7.4f} | {snr:6.2f} | {time.time()-t:6.1f}')

    # colleague MOGP-Vecchia, increasing n
    for pr in PER_RANGE:
        idx = pools[pr]; t = time.time()
        try:
            pred = fit_predict_mogp(W[idx], X[idx], m=M, steps=STEPS)
            rmse, snr = metrics(pred(W[val]), pred(W[dai]), Yv, Ya, A, dai, dac)
            log(f'{"MOGP-Vecchia":>16} | {len(idx):7d} | {rmse:7.4f} | {snr:6.2f} | {time.time()-t:6.1f}')
        except Exception as e:
            log(f'{"MOGP-Vecchia":>16} | {len(idx):7d} | FAILED {type(e).__name__}: {str(e)[:60]}')
    log('=' * 60); log('done')


if __name__ == '__main__':
    main()
