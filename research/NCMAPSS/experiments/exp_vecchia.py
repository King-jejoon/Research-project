"""
exp_vecchia.py  —  Does the Vecchia GP healthy baseline improve on the current
ExactGP-on-180-points baseline?

Compares, on identical held-out metrics from exp_lib:
  * BASELINE : MultitaskGPModel (rank-2 ICM + RBF), ExactGP, fit on dev_normal_idx (180 pts)
  * VECCHIA-180 : per-sensor Vecchia, fit on the SAME 180 pts        (isolates approximation)
  * VECCHIA-BIG : per-sensor Vecchia, fit on ~N_big normal pts       (isolates extra data)

Metrics (lower RMSE/NLL = better baseline; higher SNR = cleaner degradation signal):
  - held-out healthy RMSE / Gaussian NLL on an independent normal pool
  - degradation SNR (tail |residual| / healthy residual std) on DEV abnormal trajectories

Run:
  /opt/anaconda3/envs/pt_prac/bin/python3 exp_vecchia.py            # full
  /opt/anaconda3/envs/pt_prac/bin/python3 exp_vecchia.py --smoke    # tiny, fast sanity
"""
import os, sys, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch, gpytorch
import exp_lib as L                       # adds ROOT to path, loads cache, metrics
from vecchia_gp import VecchiaGP


# ---- fair control: ExactGP with an ARD RBF kernel (per-dim lengthscales) ----
class MultitaskGPModelARD(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        d = train_x.shape[1]
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=d), num_tasks=num_tasks, rank=2)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def train_gp_ard(W, X_s, train_idx, device='cpu', iters=100, lr=0.1, seed=0):
    L.set_seed(seed)
    x = torch.tensor(W[train_idx], dtype=torch.float32, device=device)
    y = torch.tensor(X_s[train_idx], dtype=torch.float32, device=device)
    x_mean, x_std = x.mean(0), x.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    xs = (x - x_mean) / x_std; ys = (y - y_mean) / y_std
    nt = ys.shape[1]
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nt).to(device)
    model = MultitaskGPModelARD(xs, ys, lik, nt).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.train(); lik.train()
    for i in range(iters):
        opt.zero_grad(); loss = -mll(model(xs), ys); loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik, L.GPScaler(x_mean, x_std, y_mean, y_std)

OUT = L.OUTPUT_NAMES
HI_SENSORS = ['T48', 'T50', 'Wf']         # sensors used downstream by the neural HI


# ---- numpy metric helpers (mirror exp_lib.healthy_fit_metrics) ----
def fit_metrics_from_pred(mean, std, y):
    err = y - mean
    rmse = float(np.sqrt((err ** 2).mean()))
    rmse_per = np.sqrt((err ** 2).mean(0))
    var = std ** 2 + 1e-12
    nll = float((0.5 * (np.log(2 * np.pi * var) + err ** 2 / var)).mean())
    return {'rmse': rmse, 'rmse_per': rmse_per, 'nll': nll}


def traj_from_pred(mean, X_s, A, abn_indices, abn_cycles):
    y = X_s[abn_indices]
    res = y - mean
    units = A[abn_indices, 0].astype(int)
    cycles = abn_cycles.astype(int)
    out = {}
    for u in np.unique(units):
        um = units == u
        out[int(u)] = {}
        u_cyc = cycles[um]; u_res = res[um]
        uniq = np.unique(u_cyc)
        for j, name in enumerate(OUT):
            out[int(u)][name] = (uniq, np.array([u_res[u_cyc == c, j].mean() for c in uniq]))
    return out


def healthy_std_from_pred(mean, X_s, val_idx):
    res = X_s[val_idx] - mean
    return {name: float(res[:, j].std()) for j, name in enumerate(OUT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--per_range_big', type=int, default=300)   # ~9*300 = 2700 normal pts
    ap.add_argument('--per_range_val', type=int, default=150)
    ap.add_argument('--m', type=int, default=15)
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--gp_iters', type=int, default=100)
    args = ap.parse_args()
    if args.smoke:
        args.per_range_big, args.per_range_val, args.m, args.iters, args.gp_iters = 40, 30, 8, 20, 20

    c = L.load_cache()
    W_dev, X_s_dev, A_dev = c['W_dev'], c['X_s_dev'], c['A_dev']
    normal_ranges = c['normal_ranges']
    dev_normal_idx = c['dev_normal_idx']
    dev_abn_idx, dev_abn_cyc = c['dev_abn_indices'], c['dev_abn_cycles']

    # independent held-out normal pool + a large training pool (disjoint)
    train_big = L.dense_healthy_pool(normal_ranges, per_range=args.per_range_big,
                                     seed=1, exclude=dev_normal_idx)
    val_pool = L.dense_healthy_pool(normal_ranges, per_range=args.per_range_val,
                                    seed=99, exclude=np.concatenate([train_big, dev_normal_idx]))
    print(f'[data] baseline train = {len(dev_normal_idx)} pts | '
          f'vecchia-big train = {len(train_big)} pts | held-out val = {len(val_pool)} pts')

    results = {}

    # ---------------- BASELINE : ExactGP on 180 ----------------
    t = time.time()
    model, lik, sc = L.train_gp(W_dev, X_s_dev, dev_normal_idx, iters=args.gp_iters, seed=0)
    fit = L.healthy_fit_metrics(model, lik, sc, W_dev, X_s_dev, val_pool)
    hstd, _ = L.healthy_residual_std(model, lik, sc, W_dev, X_s_dev, val_pool)
    traj = L.residual_trajectory(model, lik, sc, W_dev, X_s_dev, A_dev, dev_abn_idx, dev_abn_cyc)
    snr = L.degradation_snr(traj, hstd, OUT)
    results['ExactGP-180'] = dict(fit=fit, snr=snr, sec=time.time() - t)
    print(f'[ExactGP-180]  rmse={fit["rmse"]:.4f}  nll={fit["nll"]:.3f}  ({results["ExactGP-180"]["sec"]:.1f}s)')

    # ---------------- fair control : ExactGP-ARD on 180 ----------------
    t = time.time()
    model, lik, sc = train_gp_ard(W_dev, X_s_dev, dev_normal_idx, iters=args.gp_iters, seed=0)
    fit = L.healthy_fit_metrics(model, lik, sc, W_dev, X_s_dev, val_pool)
    hstd, _ = L.healthy_residual_std(model, lik, sc, W_dev, X_s_dev, val_pool)
    traj = L.residual_trajectory(model, lik, sc, W_dev, X_s_dev, A_dev, dev_abn_idx, dev_abn_cyc)
    snr = L.degradation_snr(traj, hstd, OUT)
    results['ExactGP-ARD-180'] = dict(fit=fit, snr=snr, sec=time.time() - t)
    print(f'[ExactGP-ARD-180]  rmse={fit["rmse"]:.4f}  nll={fit["nll"]:.3f}  ({results["ExactGP-ARD-180"]["sec"]:.1f}s)')

    # ---------------- does ExactGP scale to the big pool? ----------------
    print(f'[scaling] attempting ExactGP-ARD on {len(train_big)} pts (multitask dim '
          f'= {len(train_big)}x{len(OUT)} = {len(train_big)*len(OUT)}) ...')
    try:
        t = time.time()
        model, lik, sc = train_gp_ard(W_dev, X_s_dev, train_big, iters=args.gp_iters, seed=0)
        fit = L.healthy_fit_metrics(model, lik, sc, W_dev, X_s_dev, val_pool)
        hstd, _ = L.healthy_residual_std(model, lik, sc, W_dev, X_s_dev, val_pool)
        traj = L.residual_trajectory(model, lik, sc, W_dev, X_s_dev, A_dev, dev_abn_idx, dev_abn_cyc)
        snr = L.degradation_snr(traj, hstd, OUT)
        results[f'ExactGP-ARD-{len(train_big)}'] = dict(fit=fit, snr=snr, sec=time.time() - t)
        print(f'[ExactGP-ARD-{len(train_big)}]  rmse={fit["rmse"]:.4f}  nll={fit["nll"]:.3f}  '
              f'({results[f"ExactGP-ARD-{len(train_big)}"]["sec"]:.1f}s)')
    except (RuntimeError, MemoryError) as e:
        print(f'[ExactGP-ARD-{len(train_big)}]  FAILED to scale: {type(e).__name__}: {str(e)[:90]}')

    # ---------------- VECCHIA variants ----------------
    def run_vecchia(name, train_idx):
        t = time.time()
        vg = VecchiaGP(m=args.m).fit(W_dev[train_idx], X_s_dev[train_idx],
                                     iters=args.iters, verbose=True)
        mean_val, std_val = vg.predict(W_dev[val_pool])
        fit = fit_metrics_from_pred(mean_val, std_val, X_s_dev[val_pool])
        hstd = healthy_std_from_pred(mean_val, X_s_dev, val_pool)
        mean_abn, _ = vg.predict(W_dev[dev_abn_idx])
        traj = traj_from_pred(mean_abn, X_s_dev, A_dev, dev_abn_idx, dev_abn_cyc)
        snr = L.degradation_snr(traj, hstd, OUT)
        results[name] = dict(fit=fit, snr=snr, sec=time.time() - t)
        print(f'[{name}]  rmse={fit["rmse"]:.4f}  nll={fit["nll"]:.3f}  ({results[name]["sec"]:.1f}s)')

    run_vecchia('Vecchia-180', dev_normal_idx)
    run_vecchia(f'Vecchia-{len(train_big)}', train_big)

    # ---------------- report (print + persist to file) ----------------
    lines = ['=' * 72,
             f'{"method":>18} | {"RMSE":>8} | {"NLL":>8} | {"meanSNR":>8} | {"SNR(HI3)":>8} | {"sec":>6}',
             '-' * 72]
    for name, r in results.items():
        msnr = np.nanmean([r['snr'][s] for s in OUT])
        hsnr = np.nanmean([r['snr'][s] for s in HI_SENSORS])
        lines.append(f'{name:>18} | {r["fit"]["rmse"]:8.4f} | {r["fit"]["nll"]:8.3f} | '
                     f'{msnr:8.3f} | {hsnr:8.3f} | {r["sec"]:6.1f}')
    lines.append('=' * 72)
    table = '\n'.join(lines)
    print('\n' + table)
    with open(os.path.join(HERE, 'vecchia_results.txt'), 'w') as f:
        f.write(f'config: per_range_big={args.per_range_big} m={args.m} '
                f'iters={args.iters} gp_iters={args.gp_iters}\n')
        f.write(f'baseline_train={len(dev_normal_idx)} big_train={len(train_big)} '
                f'val={len(val_pool)}\n\n')
        f.write(table + '\n')

    # ---------------- plot ----------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names = list(results.keys())
        rmse = [results[n]['fit']['rmse'] for n in names]
        nll = [results[n]['fit']['nll'] for n in names]
        msnr = [np.nanmean([results[n]['snr'][s] for s in OUT]) for n in names]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, vals, ttl, lo in zip(axes, [rmse, nll, msnr],
                                     ['Held-out healthy RMSE  (lower better)',
                                      'Held-out healthy NLL  (lower better)',
                                      'Degradation SNR  (higher better)'],
                                     [True, True, False]):
            colors = ['#9aa7b4', '#5d89bd', '#2e8b7f'][:len(names)]
            ax.bar(names, vals, color=colors)
            ax.set_title(ttl, fontsize=10)
            ax.tick_params(axis='x', rotation=20)
            ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        p = os.path.join(HERE, 'vecchia_comparison.png')
        plt.savefig(p, dpi=160)
        print('saved', p)
    except Exception as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()
