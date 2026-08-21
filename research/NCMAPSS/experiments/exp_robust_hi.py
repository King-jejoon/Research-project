"""
Stage B 오염이 '최종 HI 곡선'까지 미치는 영향 (end-to-end).
clean-mean HI 를 기준으로, 오염 상태에서 mean vs median 집계의 HI가 얼마나 틀어지는지.
기존 sensor_processing / neural_fusion 호출.
"""
import os, sys
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L
from exp_robust import build_traj, agg_mean, agg_median, SENSORS
from sensor_processing import select_and_normalize
from neural_fusion import train_model, predict

EPOCHS = 600


def traj_to_nested(traj):
    out = {}
    for u, cyc_d in traj.items():
        out[u] = {s: {} for s in SENSORS}
        for cyc, vec in cyc_d.items():
            for s in SENSORS:
                out[u][s][int(cyc)] = float(vec[L.OUTPUT_NAMES.index(s)])
    return out


def make_hi(nested, seed=0):
    res = select_and_normalize(nested, nested, SENSORS)
    dn = res['dev_normalized']
    eng = []
    for unit in sorted(dn.keys()):
        ud = dn[unit]; cycles = sorted(next(iter(ud.values())).keys())
        eng.append(np.array([[ud[s][cyc] for s in SENSORS] for cyc in cycles]))
    np.random.seed(seed)
    nn, _ = train_model(eng, epochs=EPOCHS, lambda0=1.0, lambda1=6.0, lambda2=2.0,
                        init_threshold=0.2, alpha=0.001, verbose=False)
    return [np.asarray(h).flatten() for h in predict(nn, eng)]


def compare(his_ref, his):
    devs, finals, trends = [], [], []
    for a, b in zip(his_ref, his):
        devs.append(np.abs(a - b).mean())
        finals.append(abs(a[-1] - b[-1]))
        trends.append(stats.spearmanr(np.arange(len(b)), b)[0])
    return np.mean(devs), np.mean(finals), np.nanmean(trends)


def main():
    c = L.load_cache()
    W, Xc, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    abn_i, abn_c = c['dev_abn_indices'], c['dev_abn_cycles']
    model, lik, sc = L.train_gp(W, Xc, c['dev_normal_idx'], iters=100, seed=0)
    mean_pred, _ = L.gp_predict(model, lik, sc, W, abn_i)
    res_clean = Xc[abn_i] - mean_pred
    units = A[abn_i, 0].astype(int); cyc = abn_c.astype(int)
    res_std = res_clean.std(0)

    # 기준: 오염 없는 mean HI
    print('clean-mean HI 생성...', flush=True)
    hi_ref = make_hi(traj_to_nested(build_traj(res_clean, units, cyc, agg_mean)))

    p, k = 0.2, 6.0
    rng = np.random.default_rng(0)
    res_cont = res_clean.copy()
    n = int(len(res_cont) * p)
    hit = rng.choice(len(res_cont), n, replace=False)
    sgn = rng.choice([-1, 1], size=(n, res_cont.shape[1]))
    res_cont[hit] = res_clean[hit] + sgn * k * res_std

    print(f'오염(p={p:.0%}) 상태에서 mean / median HI 생성...', flush=True)
    hi_cont_mean = make_hi(traj_to_nested(build_traj(res_cont, units, cyc, agg_mean)))
    hi_cont_med = make_hi(traj_to_nested(build_traj(res_cont, units, cyc, agg_median)))

    print('\n' + '=' * 70)
    print(f'오염(p={p:.0%}) HI가 clean-mean HI에서 얼마나 틀어지나 (작을수록 robust)')
    print('=' * 70)
    for label, his in [('오염+mean(baseline)', hi_cont_mean), ('오염+median(robust)', hi_cont_med)]:
        d, f, tr = compare(hi_ref, his)
        print(f'  {label:22s} 곡선편차={d:.4f}  최종HI편차={f:.4f}  trend(rho)={tr:.4f}')
    dm = compare(hi_ref, hi_cont_mean)[0]; dd = compare(hi_ref, hi_cont_med)[0]
    print(f'\n  → median이 곡선편차를 {(dm-dd)/dm*100:.1f}% 줄임 (오염이 HI까지 가는 걸 차단)')


if __name__ == '__main__':
    main()
