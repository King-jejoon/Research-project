"""
최종 검증: 학습점 선택(baseline vs kcenter)이 '실제 HI 곡선 품질'까지 개선하는가?
기존 neural_fusion / sensor_processing 을 호출해서 끝까지 연결한다.

HI 품질 지표:
  - mono_viol  : HI 단조 증가 위반 비율(낮을수록 좋음)
  - trend(rho) : cycle vs HI Spearman 상관(1에 가까울수록 깨끗한 열화추세)
  - smooth     : |2차 차분| 평균(낮을수록 매끈)
  - HI 강건성  : GP 선택 seed별 HI 곡선의 변동(낮을수록 랜덤성에 불변)
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L
from run_experiments import kcenter_select
from sensor_processing import select_and_normalize   # 기존 코드 호출
from neural_fusion import train_model, predict        # 기존 코드 호출
from scipy import stats

SENSORS = ['T48', 'T50', 'Wf']
DEVICE = 'cpu'
EPOCHS = 600


def traj_to_nested(traj):
    """residual_trajectory(dict[unit][sensor]=(cyc,means)) → dict[unit][sensor][cycle]=val."""
    out = {}
    for u, d in traj.items():
        out[u] = {}
        for name, (cyc, means) in d.items():
            out[u][name] = {int(c): float(m) for c, m in zip(cyc, means)}
    return out


def build_engine_list(dev_normalized):
    lst = []
    for unit in sorted(dev_normalized.keys()):
        ud = dev_normalized[unit]
        cycles = sorted(next(iter(ud.values())).keys())
        data = np.array([[ud[name][cyc] for name in SENSORS] for cyc in cycles])
        lst.append(data)
    return lst


def hi_metrics(his):
    mono, trend, smooth = [], [], []
    for hi in his:
        hi = np.asarray(hi).flatten()
        if len(hi) < 5:
            continue
        d = np.diff(hi)
        mono.append((d < 0).mean())                       # 감소(위반) 비율
        trend.append(stats.spearmanr(np.arange(len(hi)), hi)[0])
        smooth.append(np.abs(np.diff(hi, 2)).mean())
    return np.mean(mono), np.nanmean(trend), np.mean(smooth)


def run_one(c, select_fn, seed):
    W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    abn_i, abn_c = c['dev_abn_indices'], c['dev_abn_cycles']
    idx = select_fn(seed)
    model, lik, sc = L.train_gp(W, X, idx, device=DEVICE, iters=100, seed=seed)
    traj = L.residual_trajectory(model, lik, sc, W, X, A, abn_i, abn_c, device=DEVICE)
    nested = traj_to_nested(traj)
    res = select_and_normalize(nested, nested, SENSORS)   # 기존 코드
    eng_list = build_engine_list(res['dev_normalized'])
    np.random.seed(seed)   # neural_fusion 가중치 초기화 고정
    nn, _ = train_model(eng_list, epochs=EPOCHS, lambda0=1.0, lambda1=6.0, lambda2=2.0,
                        init_threshold=0.2, alpha=0.001, verbose=False)
    his = predict(nn, eng_list)
    return his


def main():
    t0 = time.time()
    c = L.load_cache()
    base = c['dev_normal_idx']
    pool = L.dense_healthy_pool(c['normal_ranges'], per_range=400, seed=7)
    budget = len(base)
    W = c['W_dev']

    nr_list = [tuple(r) for r in c['normal_ranges']]
    selectors = {
        'baseline(linspace)': lambda s: base,
        'kcenter(B-1)':       lambda s: kcenter_select(W, pool, budget, seed=s),
        'random(최악사례)':    lambda s: L.random_pick(nr_list, K=budget // len(nr_list), seed=s),
    }

    seeds = [0, 1, 2, 3, 4]
    print('=' * 74)
    print(f'최종 검증 — HI 품질 (neural_fusion, epochs={EPOCHS}, seeds={seeds})')
    print('=' * 74)
    hi_store = {}
    for name, fn in selectors.items():
        all_mono, all_trend, all_smooth = [], [], []
        per_seed_curves = []   # list over seeds of (list over engines of HI arrays)
        for s in seeds:
            his = run_one(c, fn, s)
            per_seed_curves.append([np.asarray(h).flatten() for h in his])
            m, tr, sm = hi_metrics(his)
            all_mono.append(m); all_trend.append(tr); all_smooth.append(sm)
        hi_store[name] = per_seed_curves[0]
        am, at, asm = np.mean(all_mono), np.mean(all_trend), np.mean(all_smooth)
        # HI 강건성: 엔진별로 seed간 곡선 전체의 pointwise std → 평균
        n_eng = len(per_seed_curves[0])
        curve_stds, final_stds = [], []
        for e in range(n_eng):
            stack = np.stack([per_seed_curves[s][e] for s in range(len(seeds))])  # (n_seed, T)
            curve_stds.append(stack.std(axis=0).mean())
            final_stds.append(stack[:, -1].std())
        robust_curve = np.mean(curve_stds)
        robust_final = np.mean(final_stds)
        print(f'\n  [{name}]')
        print(f'    mono_viol={am:.4f}   trend(rho)={at:.4f}   smooth={asm:.5f}')
        print(f'    HI 강건성: 곡선전체 seed간 std={robust_curve:.4f}  최종HI seed간 std={robust_final:.4f}')

    # 두 방법 직접 비교
    print('\n' + '-' * 74)
    b, k = hi_store['baseline(linspace)'], hi_store['kcenter(B-1)']
    print('  seed0 엔진별 최종 HI (이상적으로 ~1.0 부근, 일관될수록 좋음):')
    bf = np.array([np.asarray(h).flatten()[-1] for h in b])
    kf = np.array([np.asarray(h).flatten()[-1] for h in k])
    print(f'    baseline final HI: mean={bf.mean():.3f} std={bf.std():.3f}')
    print(f'    kcenter  final HI: mean={kf.mean():.3f} std={kf.std():.3f}')
    print(f'\n총 소요 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
