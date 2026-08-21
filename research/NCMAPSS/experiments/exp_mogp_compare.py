"""
Formal comparison:  PREVIOUS code (ExactGP, gp_residuals.MultitaskGPModel)
                vs  COLLEAGUE code (MOGP + Vecchia, DEMO/).

Same data, same metrics (held-out healthy RMSE, degradation SNR).
NLL is reported only for ExactGP — the colleague's predict API exposes the
posterior mean but not a sensor-unit predictive variance, so a comparable NLL
is omitted (marked '-').

Methods:
  ExactGP-180         : previous code, exact, trained on dev_normal_idx (180)
  MOGP-Vecchia-180    : colleague code, SAME 180 pts (isolates the approximation)
  MOGP-Vecchia-2700   : colleague code, ~2700 normal pts (the scalability gain)

Run (env pt_prac):
  python exp_mogp_compare.py
"""
import os, sys, time
import numpy as np
import torch
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)

import exp_lib as L
from exp_vecchia import fit_metrics_from_pred, traj_from_pred, healthy_std_from_pred, OUT, HI_SENSORS
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp, vecchia_mogp_predictive_mean

DEV = 'cpu'
DTYPE = torch.float64
RANK = 2            # match ExactGP rank-2 ICM
KERNEL = 'rbf'      # match ExactGP RBF
M = 20              # Vecchia conditioning size
STEPS = 120
LR = 0.05


def fit_predict_mogp(Wtr, Ytr, m, steps):
    xs = StandardScaler().fit(Wtr)
    ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DTYPE, device=DEV)
    tY = torch.tensor(ys.transform(Ytr), dtype=DTYPE, device=DEV)
    q = Ytr.shape[1]
    model = GPyTorchMOGP(input_dim=Wtr.shape[1], num_tasks=q, rank=RANK, kernel=KERNEL).to(DEV, DTYPE)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=q).to(DEV, DTYPE)
    res = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=LR, group=True, verbose=False)
    model.eval(); lik.eval()

    def predict(Wte):
        teX = torch.tensor(xs.transform(Wte), dtype=DTYPE, device=DEV)
        with torch.no_grad():
            mu = vecchia_mogp_predictive_mean(model, lik, tX, tY, teX, res.structure, m=m)
        return mu.cpu().numpy() * ys.scale_ + ys.mean_
    return predict


def main():
    c = L.load_cache()
    W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    nr = c['normal_ranges']
    dni = c['dev_normal_idx']
    dai, dac = c['dev_abn_indices'], c['dev_abn_cycles']
    big = L.dense_healthy_pool(nr, per_range=300, seed=1, exclude=dni)
    val = L.dense_healthy_pool(nr, per_range=150, seed=99, exclude=np.concatenate([big, dni]))
    print(f'[data] baseline={len(dni)}  big={len(big)}  held-out val={len(val)}  abn={len(dai)}', flush=True)

    results = {}

    # ---------- A: previous code (ExactGP-180) ----------
    t = time.time()
    model, lik, sc = L.train_gp(W, X, dni, iters=100, seed=0)
    fit = L.healthy_fit_metrics(model, lik, sc, W, X, val)
    hstd, _ = L.healthy_residual_std(model, lik, sc, W, X, val)
    traj = L.residual_trajectory(model, lik, sc, W, X, A, dai, dac)
    snr = L.degradation_snr(traj, hstd, OUT)
    results['ExactGP-180 (prev)'] = dict(rmse=fit['rmse'], nll=fit['nll'], snr=snr, sec=time.time() - t)
    print(f"[ExactGP-180] rmse={fit['rmse']:.4f} done {results['ExactGP-180 (prev)']['sec']:.1f}s", flush=True)

    # ---------- B/C: colleague code (MOGP-Vecchia) ----------
    def run_mogp(name, idx, m, steps):
        t = time.time()
        try:
            predict = fit_predict_mogp(W[idx], X[idx], m=m, steps=steps)
            mval = predict(W[val])
            fit = fit_metrics_from_pred(mval, np.ones_like(mval), X[val])   # std dummy -> ignore nll
            hstd = healthy_std_from_pred(mval, X, val)
            mabn = predict(W[dai])
            traj = traj_from_pred(mabn, X, A, dai, dac)
            snr = L.degradation_snr(traj, hstd, OUT)
            results[name] = dict(rmse=fit['rmse'], nll=None, snr=snr, sec=time.time() - t)
            print(f"[{name}] rmse={fit['rmse']:.4f} done {results[name]['sec']:.1f}s", flush=True)
        except Exception as e:
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:120]}", flush=True)

    run_mogp('MOGP-Vecchia-180', dni, m=min(M, len(dni) - 1), steps=STEPS)
    run_mogp(f'MOGP-Vecchia-{len(big)}', big, m=M, steps=STEPS)

    # ---------- report ----------
    lines = ['=' * 78,
             f'{"method":>22} | {"RMSE":>8} | {"NLL":>8} | {"meanSNR":>8} | {"SNR(HI3)":>8} | {"sec":>6}',
             '-' * 78]
    for name, r in results.items():
        msnr = np.nanmean([r['snr'][s] for s in OUT])
        hsnr = np.nanmean([r['snr'][s] for s in HI_SENSORS])
        nll = f"{r['nll']:.3f}" if r['nll'] is not None else '   -'
        lines.append(f'{name:>22} | {r["rmse"]:8.4f} | {nll:>8} | {msnr:8.3f} | {hsnr:8.3f} | {r["sec"]:6.1f}')
    lines.append('=' * 78)
    table = '\n'.join(lines)
    print('\n' + table, flush=True)
    with open(os.path.join(HERE, 'mogp_compare_results.txt'), 'w') as f:
        f.write(f'config: rank={RANK} kernel={KERNEL} m={M} steps={STEPS} lr={LR}\n')
        f.write('ExactGP = previous code (gp_residuals.MultitaskGPModel); '
                'MOGP-Vecchia = colleague code (DEMO/)\n\n')
        f.write(table + '\n')
    print('saved mogp_compare_results.txt', flush=True)


if __name__ == '__main__':
    main()
