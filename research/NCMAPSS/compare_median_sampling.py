"""
GPR 품질 비교: Baseline vs Representative Index Sampling

평가 지표: GPR 예측 RMSE (dev abnormal 구간)
  - y_true  : 이상 구간의 실제 센서값
  - y_pred  : GPR이 예측한 센서값
  - RMSE    : sqrt(mean((y_true - y_pred)^2))  센서별 + 전체 평균
"""

import sys
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from load_ncmapss       import load_ncmapss
from preprocess_ncmapss import preprocess_ncmapss
from sampling_ncmapss   import run_sampling
from gp_residuals       import MultitaskGPModel

import gpytorch

# ─────────────────────────────────────────────
# 하이퍼파라미터
# ─────────────────────────────────────────────
FILENAME     = '/Users/a1/Desktop/project/pytorch_practice/data/N-CMAPSS_DS03-012.h5'
K            = 20
N_ABN        = 10
HALF_WINS    = [20, 100, 500, 1000, 2000]   # 비교할 window 크기들
TOP_K_INIT   = 10
TOP_K_STEP   = 5
GPR_ITERS    = 100
GPR_LR       = 0.1
SEED         = 42

OUTPUT_NAMES = ['T24','T30','T48','T50','P15','P2','P21','P24',
                'Ps30','P40','P50','Nf','Nc','Wf']


# ─────────────────────────────────────────────
# 대표 인덱스 탐색
# ─────────────────────────────────────────────
def find_representative_indices(W, X_s, sampled_idx, A_context,
                                normal_ranges,
                                half_win=20, top_k_init=10, top_k_step=5):
    """
    window는 반드시 해당 인덱스가 속한 normal_range 안으로 제한.
    엔진 경계 + 정상/이상 구간 경계 모두 준수.
    """
    # 이진탐색용 normal range 배열
    range_starts = np.array([s for (s, _) in normal_ranges])
    range_ends   = np.array([e for (_, e) in normal_ranges])

    rep_indices = np.empty(len(sampled_idx), dtype=int)

    for k, i in enumerate(sampled_idx):
        # i가 속한 normal range 찾기
        ri = np.searchsorted(range_starts, i, side='right') - 1
        if ri >= 0 and range_starts[ri] <= i <= range_ends[ri]:
            range_lo = range_starts[ri]
            range_hi = range_ends[ri]
        else:
            # 해당 range를 못 찾으면 원본 그대로
            rep_indices[k] = i
            continue

        # window를 normal range 경계 안으로 클리핑
        lo    = max(range_lo, i - half_win)
        hi    = min(range_hi, i + half_win)
        w_idx = np.arange(lo, hi + 1)

        x_win = W[w_idx]
        y_win = X_s[w_idx]

        x_med = np.median(x_win, axis=0)
        y_med = np.median(y_win, axis=0)

        x_std = x_win.std(axis=0) + 1e-8
        y_std = y_win.std(axis=0) + 1e-8

        x_dist = np.sqrt(((x_win - x_med) / x_std) ** 2).sum(axis=1)
        y_dist = np.sqrt(((y_win - y_med) / y_std) ** 2).sum(axis=1)

        x_rank = np.argsort(x_dist)
        y_rank = np.argsort(y_dist)

        top_k    = top_k_init
        max_k    = len(w_idx)
        chosen_j = None

        while top_k <= max_k:
            common = set(x_rank[:top_k].tolist()) & set(y_rank[:top_k].tolist())
            if common:
                common_global = w_idx[list(common)]
                chosen_j = common_global[np.argmin(np.abs(common_global - i))]
                break
            top_k += top_k_step

        rep_indices[k] = chosen_j if chosen_j is not None else i

    return rep_indices


# ─────────────────────────────────────────────
# GPR 학습 + RMSE 계산
# ─────────────────────────────────────────────
def run_gpr_and_rmse(use_idx, dev_abnormal_idx,
                     W_dev, X_s_dev, W_test, X_s_test,
                     device):
    # ── 학습 데이터 구성 ──
    x_train_np = W_dev[use_idx].astype(np.float32)
    y_train_np = X_s_dev[use_idx].astype(np.float32)

    x_train = torch.tensor(x_train_np).to(device)
    y_train = torch.tensor(y_train_np).to(device)

    x_mean = x_train.mean(0); x_std = x_train.std(0)
    y_mean = y_train.mean(0); y_std = y_train.std(0)

    x_train_sc = (x_train - x_mean) / x_std
    y_train_sc = (y_train - y_mean) / y_std

    # ── 이상 구간 입력 스케일링 ──
    x_abn_np = W_dev[dev_abnormal_idx['indices']].astype(np.float32)
    x_abn_sc = ((torch.tensor(x_abn_np) - x_mean) / x_std).to(device)

    # ── GPR 학습 ──
    num_tasks  = y_train_sc.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
    model      = MultitaskGPModel(x_train_sc, y_train_sc, likelihood, num_tasks).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=GPR_LR)
    mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    model.train(); likelihood.train()
    for it in range(GPR_ITERS):
        optimizer.zero_grad()
        loss = -mll(model(x_train_sc), y_train_sc)
        loss.backward()
        optimizer.step()
        if (it + 1) % 10 == 0:
            print(f"    Iter {it+1}/{GPR_ITERS}  loss={loss.item():.4f}")

    # ── 예측 ──
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds = likelihood(model(x_abn_sc))
        y_pred_sc = preds.mean   # (N_abn, 14)

    # 역정규화
    y_pred = (y_pred_sc * y_std + y_mean).cpu().numpy()   # (N_abn, 14)
    y_true = X_s_dev[dev_abnormal_idx['indices']]          # (N_abn, 14)

    # ── 센서별 RMSE ──
    rmse_per_sensor = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))  # (14,)
    rmse_overall    = float(np.sqrt(((y_true - y_pred) ** 2).mean()))

    return rmse_per_sensor, rmse_overall


# ─────────────────────────────────────────────
# 결과 출력
# ─────────────────────────────────────────────
def print_rmse_table(labels, all_rmse, all_overall):
    cw   = 10
    base = all_rmse[0]
    base_ov = all_overall[0]

    print(f"\n{'='*(8 + cw*len(labels) + 20)}")
    print(f"  GPR RMSE 비교 (dev abnormal 구간)  ▲=개선 ▼=저하")
    print(f"{'='*(8 + cw*len(labels) + 20)}")

    header = f"{'센서':<8}" + "".join(f"{l:>{cw}}" for l in labels)
    print(header)
    print("-" * (8 + cw * len(labels) + 20))

    for j, name in enumerate(OUTPUT_NAMES):
        line = f"{name:<8}"
        for k, rmse in enumerate(all_rmse):
            v    = rmse[j]
            mark = ''
            if k > 0:
                diff = v - base[j]
                mark = '▲' if diff < 0 else '▼'
            line += f"{v:>{cw-1}.4f}{mark if k>0 else ' '}"
        print(line)

    print("-" * (8 + cw * len(labels) + 20))
    line = f"{'[전체]':<8}"
    for k, ov in enumerate(all_overall):
        mark = ''
        if k > 0:
            diff = ov - base_ov
            pct  = diff / (base_ov + 1e-10) * 100
            mark = f"({'▲' if diff<0 else '▼'}{abs(pct):.2f}%)"
        line += f"{ov:>{cw-1}.4f} "
        if k > 0:
            line = line.rstrip() + f"{mark:<10}"
    print(line)
    print(f"{'='*(8 + cw*len(labels) + 20)}")


def save_rmse_plot_multi(labels, all_rmse, out_path):
    x      = np.arange(len(OUTPUT_NAMES))
    n      = len(labels)
    width  = 0.8 / n
    colors = plt.cm.tab10(np.linspace(0, 0.6, n))

    fig, ax = plt.subplots(figsize=(14, 5))
    for k, (label, rmse) in enumerate(zip(labels, all_rmse)):
        offset = (k - n/2 + 0.5) * width
        ax.bar(x + offset, rmse, width, label=label, alpha=0.8, color=colors[k])

    ax.set_xticks(x); ax.set_xticklabels(OUTPUT_NAMES, rotation=45)
    ax.set_ylabel('RMSE')
    ax.set_title('GPR RMSE per Sensor — Window Size Comparison')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100); plt.close()
    print(f"  RMSE 비교 그래프 저장: {out_path}")


# ─────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("데이터 로딩 중...")
    raw = load_ncmapss(FILENAME)

    W_dev    = raw['W_dev'];    X_s_dev  = raw['X_s_dev'];  A_dev  = raw['A_dev']
    W_test   = raw['W_test'];   X_s_test = raw['X_s_test']; A_test = raw['A_test']

    np.random.seed(SEED); torch.manual_seed(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── 전처리 & 샘플링 ──
    prep          = preprocess_ncmapss(A_dev, A_test)
    normal_ranges = prep['normal_ranges']
    samp = run_sampling(A_dev, A_test, prep['eng_dev'],
                        normal_ranges,
                        prep['dev_all_ranges'],
                        prep['test_all_ranges'],
                        K=K, N=N_ABN)

    dev_normal_idx    = samp['dev_normal_idx']
    dev_abnormal_idx  = samp['dev_abnormal_idx']
    test_abnormal_idx = samp['test_abnormal_idx']

    out_dir = os.path.dirname(__file__)

    # ── Baseline GPR ──
    print("\n[Baseline] GPR 학습 중...")
    rmse_base, overall_base = run_gpr_and_rmse(
        dev_normal_idx, dev_abnormal_idx,
        W_dev, X_s_dev, W_test, X_s_test, device
    )

    # ── Proposed GPR (window 크기별) ──
    all_labels   = ['Baseline']
    all_overall  = [overall_base]
    all_rmse     = [rmse_base]

    for hw in HALF_WINS:
        print(f"\n[Proposed ±{hw}] 대표 인덱스 탐색 중...")
        np.random.seed(SEED); torch.manual_seed(SEED)
        rep_idx = find_representative_indices(
            W_dev, X_s_dev, dev_normal_idx, A_dev,
            normal_ranges=normal_ranges,
            half_win=hw, top_k_init=TOP_K_INIT, top_k_step=TOP_K_STEP
        )
        changed = (rep_idx != dev_normal_idx).sum()
        print(f"  {len(dev_normal_idx)}개 중 {changed}개 대체 ({changed/len(dev_normal_idx)*100:.1f}%)")

        print(f"[Proposed ±{hw}] GPR 학습 중...")
        rmse_p, overall_p = run_gpr_and_rmse(
            rep_idx, dev_abnormal_idx,
            W_dev, X_s_dev, W_test, X_s_test, device
        )
        all_labels.append(f'±{hw}')
        all_overall.append(overall_p)
        all_rmse.append(rmse_p)

    # ── 결과 출력 ──
    print_rmse_table(all_labels, all_rmse, all_overall)
    save_rmse_plot_multi(all_labels, all_rmse,
                         os.path.join(out_dir, 'gpr_rmse_comparison.png'))
