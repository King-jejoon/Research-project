"""
ICM coregionalization rank 비교 실험 (rank = 2, 3, 4).
기존 코드는 수정하지 않는다. gp_residuals.MultitaskGPModel 은 rank=2 가 하드코딩이라,
여기서만 rank 를 인자로 받는 동일 구조의 모델을 새로 정의해 비교한다.

평가지표 (exp_lib 와 동일):
  - held-out 정상 적합도 RMSE (낮을수록 좋음)  : 건강 baseline 재구성 품질
  - Gaussian NLL (낮을수록 좋음)
  - normalised RMSE (센서별 std 로 정규화한 평균) : 센서 스케일 편향 제거
seed 5개 평균±표준편차로 robust 비교. GPR 자체 성능만 본다.
"""
import os, sys
import numpy as np
import torch
import gpytorch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L

RANKS = [2, 3, 4]
SEEDS = [0, 1, 2, 3, 4]
PER_RANGE_VAL = 300        # held-out 검증풀: 정상구간당 점 수
ITERS = 100
LR = 0.1
DEVICE = 'cpu'


# rank 를 인자로 받는 MOGP (기존 MultitaskGPModel 과 구조 동일, rank 만 가변)
class MultitaskGPModelRank(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, rank):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=rank)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(
            self.mean_module(x), self.covar_module(x))


def train_gp_rank(W, X_s, train_idx, rank, device='cpu', iters=100, lr=0.1, seed=0):
    """exp_lib.train_gp 과 동일 절차, rank 만 가변."""
    L.set_seed(seed)
    x = torch.tensor(W[train_idx], dtype=torch.float32, device=device)
    y = torch.tensor(X_s[train_idx], dtype=torch.float32, device=device)
    x_mean, x_std = x.mean(0), x.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    nt = ys.shape[1]
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=nt).to(device)
    model = MultitaskGPModelRank(xs, ys, lik, nt, rank).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.train(); lik.train()
    for _ in range(iters):
        opt.zero_grad()
        loss = -mll(model(xs), ys)
        loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik, L.GPScaler(x_mean, x_std, y_mean, y_std)


def main():
    c = L.load_cache()
    W, X_s, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    normal_ranges = [tuple(r) for r in c['normal_ranges']]
    train_idx = np.asarray(c['dev_normal_idx'], dtype=np.int64)   # 실제 K=20 reference set
    print(f"reference (training) healthy points : {len(train_idx)}")
    print(f"ranks={RANKS}  seeds={SEEDS}  iters={ITERS}  lr={LR}\n")

    res = {r: {'rmse': [], 'nll': [], 'nrmse': []} for r in RANKS}
    for seed in SEEDS:
        val_idx = L.dense_healthy_pool(normal_ranges, per_range=PER_RANGE_VAL,
                                       seed=seed, exclude=train_idx)
        std_j = X_s[val_idx].std(axis=0) + 1e-8       # 센서별 스케일 (정규화용)
        for r in RANKS:
            model, lik, sc = train_gp_rank(W, X_s, train_idx, r, DEVICE, ITERS, LR, seed)
            m = L.healthy_fit_metrics(model, lik, sc, W, X_s, val_idx, DEVICE)
            nrmse = float(np.mean(m['rmse_per'] / std_j))
            res[r]['rmse'].append(m['rmse'])
            res[r]['nll'].append(m['nll'])
            res[r]['nrmse'].append(nrmse)
            print(f"  seed {seed} | rank {r} | RMSE={m['rmse']:.4f}  "
                  f"nRMSE={nrmse:.4f}  NLL={m['nll']:.4f}  (val n={len(val_idx)})")

    print("\n==== SUMMARY  (held-out healthy, mean ± std over %d seeds) ====" % len(SEEDS))
    print(f"{'rank':>5} | {'RMSE':>16} | {'nRMSE':>16} | {'NLL':>16}")
    for r in RANKS:
        rm = np.array(res[r]['rmse']); nr = np.array(res[r]['nrmse']); nl = np.array(res[r]['nll'])
        print(f"{r:>5} | {rm.mean():7.4f} ± {rm.std():6.4f} | "
              f"{nr.mean():7.4f} ± {nr.std():6.4f} | {nl.mean():7.4f} ± {nl.std():6.4f}")

    best_rmse  = min(RANKS, key=lambda r: np.mean(res[r]['rmse']))
    best_nrmse = min(RANKS, key=lambda r: np.mean(res[r]['nrmse']))
    best_nll   = min(RANKS, key=lambda r: np.mean(res[r]['nll']))
    print(f"\nbest by RMSE : rank {best_rmse}")
    print(f"best by nRMSE: rank {best_nrmse}")
    print(f"best by NLL  : rank {best_nll}")


if __name__ == '__main__':
    main()
