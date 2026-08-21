"""
실험 공용 라이브러리.
- 캐시 로드
- 기존 GP 구조(gp_residuals.MultitaskGPModel)를 '호출'해서 학습/예측 (기존 코드 수정 X)
- 올바른 평가지표: held-out 정상 적합도(RMSE/NLL), 열화 SNR, 잔차 백색성
"""
import os, sys
import numpy as np
import torch
import gpytorch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from gp_residuals import MultitaskGPModel  # 기존 GP 구조 재사용

CACHE = os.path.join(HERE, 'cache.npz')
OUTPUT_NAMES = ['T24','T30','T48','T50','P15','P2','P21','P24',
                'Ps30','P40','P50','Nf','Nc','Wf']

_cache = None
def load_cache():
    global _cache
    if _cache is None:
        _cache = dict(np.load(CACHE, allow_pickle=False))
    return _cache


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ─────────────────────────────────────────────
# 정상 구간에서 healthy 점들을 뽑는 헬퍼들
# ─────────────────────────────────────────────
def linspace_pick(normal_ranges, eng_dev, K, ranges_engine_first=None):
    """기존 sample_ranges와 동일 논리(엔진별 range마다 시간축 linspace K개)."""
    from sampling_ncmapss import sample_ranges
    engines = list(np.unique(eng_dev).astype(int))
    idx, _ = sample_ranges(engines, [tuple(r) for r in normal_ranges], eng_dev, K=K)
    return np.asarray(idx, dtype=np.int64)


def random_pick(normal_ranges, K, seed):
    """각 range에서 무작위 K개 (randomness 비판 대상 그 자체)."""
    rng = np.random.default_rng(seed)
    out = []
    for (s, e) in normal_ranges:
        n = e - s + 1
        k = min(K, n)
        out.append(s + rng.choice(n, size=k, replace=False))
    return np.sort(np.concatenate(out)).astype(np.int64)


def dense_healthy_pool(normal_ranges, per_range, seed=0, exclude=None):
    """held-out 평가용: 정상구간에서 촘촘히 뽑은 독립 검증 풀."""
    rng = np.random.default_rng(seed)
    out = []
    for (s, e) in normal_ranges:
        n = e - s + 1
        k = min(per_range, n)
        out.append(s + rng.choice(n, size=k, replace=False))
    pool = np.unique(np.concatenate(out)).astype(np.int64)
    if exclude is not None:
        pool = np.setdiff1d(pool, np.asarray(exclude, dtype=np.int64))
    return pool


# ─────────────────────────────────────────────
# GP 학습 / 예측 (기존 구조 호출)
# ─────────────────────────────────────────────
class GPScaler:
    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std


def train_gp(W, X_s, train_idx, device='cpu', iters=100, lr=0.1, seed=0, verbose=False):
    set_seed(seed)
    x = torch.tensor(W[train_idx], dtype=torch.float32, device=device)
    y = torch.tensor(X_s[train_idx], dtype=torch.float32, device=device)
    x_mean, x_std = x.mean(0), x.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std

    num_tasks = ys.shape[1]
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
    model = MultitaskGPModel(xs, ys, lik, num_tasks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)

    model.train(); lik.train()
    for i in range(iters):
        opt.zero_grad()
        loss = -mll(model(xs), ys)
        loss.backward(); opt.step()
        if verbose and (i + 1) % 25 == 0:
            print(f'    gp iter {i+1}/{iters} loss={loss.item():.4f}')
    model.eval(); lik.eval()
    sc = GPScaler(x_mean, x_std, y_mean, y_std)
    return model, lik, sc


def train_gp_arrays(x_np, y_np, device='cpu', iters=100, lr=0.1, seed=0):
    """인덱스가 아닌 '명시적 값 배열'로 GP 학습 (A-1 윈도우 대표값 등에 사용)."""
    set_seed(seed)
    x = torch.tensor(np.asarray(x_np), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(y_np), dtype=torch.float32, device=device)
    x_mean, x_std = x.mean(0), x.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    xs = (x - x_mean) / x_std
    ys = (y - y_mean) / y_std
    num_tasks = ys.shape[1]
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
    model = MultitaskGPModel(xs, ys, lik, num_tasks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.train(); lik.train()
    for i in range(iters):
        opt.zero_grad()
        loss = -mll(model(xs), ys)
        loss.backward(); opt.step()
    model.eval(); lik.eval()
    return model, lik, GPScaler(x_mean, x_std, y_mean, y_std)


def gp_predict(model, lik, sc, W, idx, device='cpu'):
    """반정규화된 posterior 평균/표준편차(센서 단위) 반환."""
    x = torch.tensor(W[idx], dtype=torch.float32, device=device)
    xs = (x - sc.x_mean) / sc.x_std
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(model(xs))
        m = pred.mean
        v = pred.variance
    mean = (m * sc.y_std + sc.y_mean).cpu().numpy()
    std = (v.sqrt() * sc.y_std).cpu().numpy()
    return mean, std


# ─────────────────────────────────────────────
# 평가지표
# ─────────────────────────────────────────────
def healthy_fit_metrics(model, lik, sc, W, X_s, val_idx, device='cpu'):
    """held-out 정상 점에서 RMSE와 Gaussian NLL (낮을수록 좋은 baseline)."""
    mean, std = gp_predict(model, lik, sc, W, val_idx, device)
    y = X_s[val_idx]
    err = y - mean
    rmse_per = np.sqrt((err ** 2).mean(axis=0))
    rmse = float(np.sqrt((err ** 2).mean()))
    var = std ** 2 + 1e-12
    nll = 0.5 * (np.log(2 * np.pi * var) + err ** 2 / var)
    nll_mean = float(nll.mean())
    return {'rmse': rmse, 'rmse_per': rmse_per, 'nll': nll_mean}


def residual_trajectory(model, lik, sc, W, X_s, A, abn_indices, abn_cycles, device='cpu', zscore=False):
    """
    이상(전체수명) 점들에서 잔차를 구해 unit·sensor·cycle 평균 trajectory로 집계.
    zscore=True면 (y-mu)/sigma (A-1: 불확실성 가중 잔차).
    반환: dict[unit][sensor] = (cycles_sorted, mean_residual_per_cycle)
    """
    mean, std = gp_predict(model, lik, sc, W, abn_indices, device)
    y = X_s[abn_indices]
    if zscore:
        res = (y - mean) / (std + 1e-8)
    else:
        res = (y - mean)
    units = A[abn_indices, 0].astype(int)
    cycles = abn_cycles.astype(int)

    out = {}
    for u in np.unique(units):
        um = units == u
        out[int(u)] = {}
        u_cyc = cycles[um]
        u_res = res[um]
        uniq = np.unique(u_cyc)
        for j, name in enumerate(OUTPUT_NAMES):
            means = np.array([u_res[u_cyc == c, j].mean() for c in uniq])
            out[int(u)][name] = (uniq, means)
    return out


def degradation_snr(traj_by_unit, healthy_res_std, sensors):
    """
    열화 신호 대 잡음비:
      signal = 각 엔진에서 마지막 10% cycle 평균 |잔차| (열화 후기)
      noise  = healthy 잔차 std (정상 잡음 바닥)
    센서별·엔진별 평균 SNR 반환.
    """
    snrs = {}
    for name in sensors:
        vals = []
        for u, d in traj_by_unit.items():
            cyc, means = d[name]
            if len(cyc) < 10:
                continue
            tail = means[int(len(means) * 0.9):]
            sig = np.abs(tail).mean()
            vals.append(sig / (healthy_res_std[name] + 1e-8))
        snrs[name] = float(np.mean(vals)) if vals else float('nan')
    return snrs


def healthy_residual_std(model, lik, sc, W, X_s, val_idx, device='cpu', zscore=False):
    mean, std = gp_predict(model, lik, sc, W, val_idx, device)
    res = X_s[val_idx] - mean
    if zscore:
        res = res / (std + 1e-8)
    return {name: float(res[:, j].std()) for j, name in enumerate(OUTPUT_NAMES)}, res


def ordered_segment_residuals(model, lik, sc, W, X_s, seg_start, seg_end,
                              stride=50, device='cpu'):
    """한 정상 구간을 시간순서대로 잔차 계산(백색성 검정용 시계열)."""
    idx = np.arange(seg_start, seg_end + 1, stride, dtype=np.int64)
    mean, _ = gp_predict(model, lik, sc, W, idx, device)
    res = X_s[idx] - mean
    return idx, res


def ljung_box(x, lags=10):
    """수동 Ljung-Box Q 검정 (statsmodels 불필요). p가 크면 백색잡음."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    denom = (x ** 2).sum() + 1e-12
    Q = 0.0
    for k in range(1, lags + 1):
        rho = (x[k:] * x[:-k]).sum() / denom
        Q += rho ** 2 / (n - k)
    Q *= n * (n + 2)
    p = 1.0 - stats.chi2.cdf(Q, lags)
    return float(Q), float(p)
