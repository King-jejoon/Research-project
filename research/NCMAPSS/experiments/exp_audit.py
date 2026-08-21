"""
결정적 질문: '실제' 데이터에 robust가 잡아낼 오염(스파이크/이상치)이 존재하는가?
- 존재 X  → robust는 보험일 뿐, 효과 없음(현재 결과대로)
- 존재 O  → robust가 실제로 가치 있음
기존 데이터/코드 호출만.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L
from exp_robust import hampel_clean_indices, build_traj, agg_mean, agg_median, SENSORS

c = L.load_cache()
W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
nr = c['normal_ranges']
sampled = c['dev_normal_idx']

print('=' * 72)
print('A. 정상구간 실데이터의 스파이크 비율 (Hampel, 행 전체 오염 기준)')
print('=' * 72)
total_spike = 0; total_n = 0
sampled_on_spike = 0
for (s, e) in nr:
    seg = X[s:e + 1]
    keep = hampel_clean_indices(seg)
    spk = (~keep)
    total_spike += spk.sum(); total_n += len(seg)
    # 이 구간에서 샘플된 점들이 스파이크 위에 있나
    loc = sampled[(sampled >= s) & (sampled <= e)] - s
    sampled_on_spike += spk[loc].sum()
print(f'  전체 정상행 {total_n:,} 중 스파이크 {total_spike:,}  ({total_spike/total_n*100:.3f}%)')
print(f'  실제 샘플된 학습점 {len(sampled)}개 중 스파이크 위 = {sampled_on_spike}개')

print('\n' + '=' * 72)
print('B. 실데이터 잔차: cycle별 mean vs median 차이 (이상치 있으면 벌어짐)')
print('=' * 72)
model, lik, sc = L.train_gp(W, X, sampled, iters=100, seed=0)
abn_i, abn_c = c['dev_abn_indices'], c['dev_abn_cycles']
mp, _ = L.gp_predict(model, lik, sc, W, abn_i)
res = X[abn_i] - mp
units = A[abn_i, 0].astype(int); cyc = abn_c.astype(int)
cols = np.array([L.OUTPUT_NAMES.index(s) for s in SENSORS])

tr_mean = build_traj(res, units, cyc, agg_mean)
tr_med = build_traj(res, units, cyc, agg_median)
diffs = []
for u in tr_mean:
    for cc in tr_mean[u]:
        diffs.append(np.abs(tr_mean[u][cc][cols] - tr_med[u][cc][cols]))
diffs = np.array(diffs)

# cycle 내 이상치 비율 (|res-med|>3*MAD)
n_out = 0; n_tot = 0
for u in np.unique(units):
    um = units == u
    for cval in np.unique(cyc[um]):
        m = um & (cyc == cval)
        r = res[m][:, cols]
        med = np.median(r, axis=0)
        mad = np.median(np.abs(r - med), axis=0) * 1.4826 + 1e-8
        n_out += (np.abs(r - med) > 3 * mad).sum(); n_tot += r.size

print(f'  cycle별 |mean-median| 평균={diffs.mean():.4f}, 최대={diffs.max():.4f}')
print(f'    (참고: 오염 30% 주입 시 이 편차가 4.14까지 벌어졌음)')
print(f'  cycle 내 잔차 이상치(>3MAD) 비율 = {n_out/n_tot*100:.2f}%')

print('\n' + '=' * 72)
print('결론')
print('=' * 72)
sp_pct = total_spike / total_n * 100
print(f'  실데이터 스파이크율 ~{sp_pct:.3f}%, 샘플점 적중 {sampled_on_spike}개')
print(f'  실데이터 mean-median 편차 ~{diffs.mean():.4f} (오염30%일 때 4.14)')
