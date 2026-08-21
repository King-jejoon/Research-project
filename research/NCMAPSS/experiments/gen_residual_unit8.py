"""
DEV unit 8 의 센서별 잔차(residual) 추세 그림 생성.
목적: Neural Health Index Fusion 절에서 센서 부분집합 {T48, T50, Wf} 를 고른 이유를
      시각적으로 보충 설명. (열화에 따라 뚜렷하게 단조 변화하는 센서 = 좋은 열화 신호)

기존 코드는 수정하지 않고 호출만 한다:
  - exp_lib.train_gp / gp_predict  (기존 rank-2 MOGP 구조 그대로)
  - residual_analysis.compute_residuals_by_unit
스타일은 paper Figure 1(visualization.ipynb)과 동일한 결의 깔끔한 라인 플롯.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L
from residual_analysis import compute_residuals_by_unit
from gp_residuals import output_names

UNIT = 8
SELECTED = ['T48', 'T50', 'Wf']     # 융합에 실제로 쓰는 센서
SEL_COLORS = {'T48': '#1f77b4', 'T50': '#2ca02c', 'Wf': '#d62728'}
BASELINE_FRAC = 0.20                # 초기 healthy 구간 비율(노이즈 기준)
OUT_PNG = os.path.join(HERE, 'unit8_residuals.png')


def main():
    c = L.load_cache()
    W, X_s, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    train_idx = np.asarray(c['dev_normal_idx'], dtype=np.int64)
    abn_idx   = np.asarray(c['dev_abn_indices'], dtype=np.int64)
    abn_cyc   = list(np.asarray(c['dev_abn_cycles']))

    # 1) 기존 rank-2 MOGP 학습 (healthy reference 180점)
    L.set_seed(0)
    model, lik, sc = L.train_gp(W, X_s, train_idx, device='cpu', iters=100, lr=0.1, seed=0)

    # 2) full-life 평가점에서 de-normalized posterior mean -> residual = y - y_hat
    mean, _ = L.gp_predict(model, lik, sc, W, abn_idx, device='cpu')
    resid = X_s[abn_idx] - mean                          # (n, 14)

    # 3) 기존 함수로 unit/sensor/cycle 평균 잔차 집계
    dev_residuals = [{'name': output_names[j], 'dataset': 'DEV',
                      'residual': resid[:, j].copy()} for j in range(len(output_names))]
    dev_abn = {'indices': list(abn_idx), 'cycles': abn_cyc}
    R = compute_residuals_by_unit(dev_residuals, [], dev_abn,
                                  {'indices': [], 'cycles': []}, A, A)

    u = R['dev_residuals_by_unit'][UNIT]                 # {sensor: {cycle: mean_resid}}

    # 4) 센서별 cycle-residual 궤적을 '초기 healthy 노이즈(sigma)' 단위로 정규화
    #    -> 추세가 강한 센서는 sigma의 여러 배로 벗어나고, 잡음뿐인 센서는 0 근처에 머묾
    plt.figure(figsize=(7.2, 4.3))
    trend = {}      # |corr(cycle, residual)| : 추세 '모양'
    snr   = {}      # 수명말기 |sigma-deviation| : 노이즈 대비 신호 '크기'
    for name in output_names:
        cycles = np.array(sorted(u[name].keys()))
        vals   = np.array([u[name][cc] for cc in cycles], dtype=float)
        nb = max(4, int(len(cycles) * BASELINE_FRAC))
        base_mu = vals[:nb].mean()
        base_sd = vals[:nb].std() + 1e-8
        z = (vals - base_mu) / base_sd
        # 추세 강도: cycle 과의 상관(부호 무관 크기)
        trend[name] = abs(np.corrcoef(cycles, vals)[0, 1])
        # 열화 SNR: 마지막 10% 사이클의 평균 |sigma-deviation|
        ntail = max(1, int(len(z) * 0.10))
        snr[name] = float(np.abs(z[-ntail:]).mean())
        if name in SELECTED:
            plt.plot(cycles, z, color=SEL_COLORS[name], linewidth=2.2,
                     alpha=0.95, label=name, zorder=3)
        else:
            plt.plot(cycles, z, color='0.72', linewidth=1.0, alpha=0.7, zorder=1)

    plt.axhline(0, color='black', linewidth=0.8, alpha=0.6)
    plt.xlabel('Cycle', fontsize=11)
    plt.ylabel('Residual deviation [healthy $\\sigma$]', fontsize=11)
    # 범례: 선택 센서 + 회색(others) 더미
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=SEL_COLORS[s], lw=2.2, label=s) for s in SELECTED]
    handles.append(Line2D([0], [0], color='0.72', lw=1.0, label='other sensors'))
    plt.legend(handles=handles, fontsize=9, loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    print('saved:', OUT_PNG)

    # 5) 선택 근거 수치: 열화 SNR(크기) 와 추세 상관(모양)
    print('\nDEV unit %d  ranking by end-of-life degradation SNR (|sigma-dev|, last 10%%):' % UNIT)
    print(f"  {'sensor':6s} {'SNR(sigma)':>11s} {'|trend|':>8s}")
    for name in sorted(snr, key=snr.get, reverse=True):
        mark = '  <-- selected' if name in SELECTED else ''
        print(f'  {name:6s} {snr[name]:>11.1f} {trend[name]:>8.3f}{mark}')


if __name__ == '__main__':
    main()
