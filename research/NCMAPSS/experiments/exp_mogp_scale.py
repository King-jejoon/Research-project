"""
Scaling comparison: previous ExactGP (capped at 180 pts) vs colleague MOGP-Vecchia
trained on an INCREASING number of normal points (data is abundant: ~1.85M rows).

Config chosen from hardware benchmarks on this M3 Max:
  device = cpu, dtype = float32, torch threads = 6.

Results are appended to mogp_scale_results.txt after EACH level (so partial
progress survives even if the largest level is slow).

Run (env pt_prac):
  python exp_mogp_scale.py
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
from exp_vecchia import fit_metrics_from_pred, traj_from_pred, healthy_std_from_pred, OUT, HI_SENSORS
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp, vecchia_mogp_predictive_mean

DEV, DT = 'cpu', torch.float32
RANK, KERNEL, M, STEPS, LR = 2, 'rbf', 30, 250, 0.05
PER_RANGE = [100, 300, 600, 1200]          # -> ~900 / 2700 / 5400 / 10800 pts
RESULTS = os.path.join(HERE, 'mogp_scale_results.txt')


def log(line):
    print(line, flush=True)
    with open(RESULTS, 'a') as f:
        f.write(line + '\n')


def fit_predict_mogp(Wtr, Ytr, m, steps):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT, device=DEV)
    tY = torch.tensor(ys.transform(Ytr), dtype=DT, device=DEV)
    q = Ytr.shape[1]
    model = GPyTorchMOGP(Wtr.shape[1], num_tasks=q, rank=RANK, kernel=KERNEL).to(DEV, DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=q).to(DEV, DT)
    res = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=LR, group=True, verbose=False)
    model.eval(); lik.eval()
    last_ll = float(res.log_likelihoods[-1])

    def predict(Wte):
        teX = torch.tensor(xs.transform(Wte), dtype=DT, device=DEV)
        with torch.no_grad():
            mu = vecchia_mogp_predictive_mean(model, lik, tX, tY, teX, res.structure, m=m)
        return mu.cpu().numpy() * ys.scale_ + ys.mean_
    return predict, last_ll


def evaluate(predict, W, X, A, val, dai, dac):
    mval = predict(W[val])
    rmse = float(np.sqrt(((X[val] - mval) ** 2).mean()))
    hstd = healthy_std_from_pred(mval, X, val)
    mabn = predict(W[dai])
    traj = traj_from_pred(mabn, X, A, dai, dac)
    snr = L.degradation_snr(traj, hstd, OUT)
    return rmse, float(np.nanmean([snr[s] for s in OUT])), float(np.nanmean([snr[s] for s in HI_SENSORS]))


def main():
    open(RESULTS, 'w').close()
    log(f'config: device={DEV} dtype=float32 threads=6 rank={RANK} kernel={KERNEL} '
        f'm={M} steps={STEPS} lr={LR}')
    c = L.load_cache()
    W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    nr = c['normal_ranges']; dni = c['dev_normal_idx']
    dai, dac = c['dev_abn_indices'], c['dev_abn_cycles']

    pools = {pr: L.dense_healthy_pool(nr, per_range=pr, seed=1, exclude=dni) for pr in PER_RANGE}
    exclude_val = np.unique(np.concatenate([dni] + [p for p in pools.values()]))
    val = L.dense_healthy_pool(nr, per_range=150, seed=99, exclude=exclude_val)
    log(f'held-out val = {len(val)} pts | abn = {len(dai)} pts')
    log('=' * 70)
    log(f'{"method":>22} | {"n_train":>7} | {"RMSE":>7} | {"meanSNR":>7} | {"SNR_HI3":>7} | {"sec":>6}')
    log('-' * 70)

    # previous code baseline (ExactGP-180, cannot scale)
    t = time.time()
    m0, l0, sc0 = L.train_gp(W, X, dni, iters=100, seed=0)
    fit = L.healthy_fit_metrics(m0, l0, sc0, W, X, val)
    hstd, _ = L.healthy_residual_std(m0, l0, sc0, W, X, val)
    traj = L.residual_trajectory(m0, l0, sc0, W, X, A, dai, dac)
    snr = L.degradation_snr(traj, hstd, OUT)
    msnr = np.nanmean([snr[s] for s in OUT]); hsnr = np.nanmean([snr[s] for s in HI_SENSORS])
    log(f'{"ExactGP (prev)":>22} | {len(dni):7d} | {fit["rmse"]:7.4f} | {msnr:7.3f} | {hsnr:7.3f} | {time.time()-t:6.1f}')

    # colleague MOGP-Vecchia, increasing n
    for pr in PER_RANGE:
        idx = pools[pr]
        t = time.time()
        try:
            predict, ll = fit_predict_mogp(W[idx], X[idx], m=M, steps=STEPS)
            rmse, msnr, hsnr = evaluate(predict, W, X, A, val, dai, dac)
            log(f'{"MOGP-Vecchia":>22} | {len(idx):7d} | {rmse:7.4f} | {msnr:7.3f} | {hsnr:7.3f} | {time.time()-t:6.1f}')
        except Exception as e:
            log(f'{"MOGP-Vecchia":>22} | {len(idx):7d} | FAILED {type(e).__name__}: {str(e)[:70]}')
    log('=' * 70)
    log('done')


if __name__ == '__main__':
    main()
