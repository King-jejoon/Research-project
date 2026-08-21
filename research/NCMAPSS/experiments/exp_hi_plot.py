"""
디펜스용 핵심 그림 2장 생성:
 (1) 완전 무작위 GP 학습점 선택을 5개 seed로 바꿔도 HI 곡선이 거의 겹침 → 랜덤성에 둔감
 (2) 랜덤성 전파 비교: GP baseline 적합 변동(±7%) vs 최종 HI 변동(~0.7%) → ~10x 감쇠
기존 코드 호출만. 출력: experiments/fig_hi_robustness.png
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from exp_hi import run_one

SEEDS = [0, 1, 2, 3, 4]


def main():
    c = L.load_cache()
    nr_list = [tuple(r) for r in c['normal_ranges']]
    base = c['dev_normal_idx']
    budget = len(base)

    def random_sel(s):
        return L.random_pick(nr_list, K=budget // len(nr_list), seed=s)

    # 5개 seed로 HI 곡선 수집(완전 무작위 선택)
    curves = []  # per seed: list of engine HI arrays
    for s in SEEDS:
        his = run_one(c, random_sel, s)
        curves.append([np.asarray(h).flatten() for h in his])
    n_eng = len(curves[0])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # (1) 대표 엔진 3개의 HI 곡선을 seed별로 오버레이
    ax = axes[0]
    show_eng = [0, n_eng // 2, n_eng - 1]
    colors = ['tab:blue', 'tab:green', 'tab:red']
    for ci, e in enumerate(show_eng):
        for s in range(len(SEEDS)):
            hi = curves[s][e]
            ax.plot(np.arange(len(hi)), hi, color=colors[ci], alpha=0.45, lw=1.2,
                    label=f'Engine {e+1}' if s == 0 else None)
    ax.set_title('Fully-random GP training-point selection x 5 seeds\n'
                 '-> HI curves nearly coincide (insensitive to randomness)')
    ax.set_xlabel('Cycle index'); ax.set_ylabel('Health Index')
    ax.legend(); ax.grid(alpha=0.3)

    # (2) 랜덤성 전파 비교 (CV %)
    ax = axes[1]
    # GP baseline RMSE 변동(앞 실험 Exp C random): 평균 0.9845, std 0.0698 → CV
    gp_rmse_mean, gp_rmse_std = 0.9845, 0.0698
    gp_cv = gp_rmse_std / gp_rmse_mean * 100
    # HI 변동: 엔진별 곡선 pointwise std / HI 범위
    hi_rng = np.mean([np.ptp(np.concatenate([curves[s][e] for s in range(len(SEEDS))]))
                      for e in range(n_eng)])
    hi_std = np.mean([np.stack([curves[s][e] for s in range(len(SEEDS))]).std(0).mean()
                      for e in range(n_eng)])
    hi_cv = hi_std / hi_rng * 100
    bars = ax.bar(['GP baseline fit\n(held-out RMSE)', 'Final HI curve'],
                  [gp_cv, hi_cv], color=['tab:orange', 'tab:blue'], alpha=0.85)
    for b, v in zip(bars, [gp_cv, hi_cv]):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}%', ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    ax.set_ylabel('seed-to-seed CV (%)')
    ax.set_title(f'Randomness propagation: ~{gp_cv/max(hi_cv,1e-6):.0f}x attenuation\n'
                 '(GP-level wobble barely reaches the HI)')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(HERE, 'fig_hi_robustness.png')
    plt.savefig(out, dpi=110); plt.close()
    print(f'saved -> {out}')
    print(f'GP baseline RMSE CV = {gp_cv:.2f}%   HI CV = {hi_cv:.2f}%   '
          f'감쇠율 = {gp_cv/max(hi_cv,1e-6):.1f}x')
    print(f'(HI 범위 평균={hi_rng:.3f}, HI seed간 std 평균={hi_std:.4f})')


if __name__ == '__main__':
    main()
