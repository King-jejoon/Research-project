"""
추천 robust 방법들을 순서대로 구현하고, 오염 주입(contamination) 검증으로
기존 방식 대비 개선을 분석한다. 기존 코드는 호출만 한다.

핵심: clean RMSE만 보면 robust 효과가 안 보인다(이전 repidx 사례).
     → 인위적 스파이크/이상치를 주입한 뒤 '덜 망가지는가'로 평가한다.

Stage A (학습점 선택 robust):  baseline(linspace) vs A-1(윈도우 median) vs A-2(Hampel+linspace)
        + A-3(steady-state) 는 transient 위치 스파이크로 별도 검증
Stage B (잔차 집계 robust):    mean(baseline) vs median / trimmed / MAD-reject
"""
import os, sys, time
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L

DEVICE = 'cpu'
SENSORS = ['T48', 'T50', 'Wf']


# ─────────────────────────────────────────────
# 오염 주입
# ─────────────────────────────────────────────
def spike_rows(Xwork, Xclean, row_pool, frac, k_sigma, seed, std):
    """Xwork(공유 버퍼)에 in-place 스파이크 주입. 적용된 행 인덱스 반환(복원용)."""
    rng = np.random.default_rng(seed)
    n = int(len(row_pool) * frac)
    if n == 0:
        return np.array([], dtype=int)
    hit = rng.choice(row_pool, size=n, replace=False)
    sgn = rng.choice([-1, 1], size=(n, Xwork.shape[1])).astype(np.float32)
    Xwork[hit] = Xclean[hit] + sgn * k_sigma * std
    return hit


def restore_rows(Xwork, Xclean, hit):
    if len(hit):
        Xwork[hit] = Xclean[hit]


# ─────────────────────────────────────────────
# Stage A 선택 전략
# ─────────────────────────────────────────────
def hampel_clean_indices(series_2d, window=51, n_sigma=6.0, min_cols=7):
    """롤링 median±n_sigma·MAD 검사(벡터화). 여러 센서가 동시에 벗어날 때만 스파이크로 간주.
    (행 전체가 오염된 bad snapshot에 특화 → clean 오탐 최소화)"""
    from scipy import ndimage
    x = np.asarray(series_2d, dtype=np.float32)
    med = ndimage.median_filter(x, size=(window, 1), mode='nearest')
    absdev = np.abs(x - med)
    mad = ndimage.median_filter(absdev, size=(window, 1), mode='nearest') * 1.4826 + 1e-8
    cnt = (absdev > n_sigma * mad).sum(axis=1)
    spike = cnt >= min_cols
    return ~spike


def linspace_in_pool(pool_global, K):
    """pool(전역 인덱스) 안에서 시간축 균등 K개."""
    if len(pool_global) <= K:
        return pool_global
    pos = np.round(np.linspace(0, len(pool_global) - 1, K)).astype(int)
    return pool_global[np.unique(pos)]


def select_baseline(c, Xcorr):
    return c['dev_normal_idx']  # 기존 linspace 그대로


def select_A1_window_median(c, Xcorr, half_win=40):
    """균등 인덱스는 유지, 각 점의 (W,X) 값을 ±half_win 윈도우 median으로 치환."""
    W = c['W_dev']; A = c['A_dev']
    nr = c['normal_ranges']
    idx = c['dev_normal_idx']
    rs = np.array([s for s, _ in nr]); re = np.array([e for _, e in nr])
    Wv = np.empty((len(idx), W.shape[1]), np.float32)
    Xv = np.empty((len(idx), Xcorr.shape[1]), np.float32)
    for k, i in enumerate(idx):
        ri = np.searchsorted(rs, i, side='right') - 1
        lo = max(int(rs[ri]), i - half_win); hi = min(int(re[ri]), i + half_win)
        Wv[k] = np.median(W[lo:hi + 1], axis=0)
        Xv[k] = np.median(Xcorr[lo:hi + 1], axis=0)
    return ('xy', Wv, Xv)


def select_A2_hampel(c, Xcorr, K=20):
    """각 정상구간에서 Hampel로 스파이크 제거 후 균등 샘플링."""
    nr = c['normal_ranges']
    out = []
    for (s, e) in nr:
        seg = Xcorr[s:e + 1]
        keep = hampel_clean_indices(seg, window=25, n_sigma=3.0)
        pool = np.arange(s, e + 1)[keep]
        if len(pool) == 0:
            pool = np.arange(s, e + 1)
        out.append(linspace_in_pool(pool, K))
    return np.sort(np.concatenate(out)).astype(np.int64)


def fit_and_eval(c, sel_result, val_idx, Xtrain_src, seed):
    """sel_result가 인덱스면 그 행으로, ('xy',W,X)면 명시 값으로 GP 학습 → clean held-out RMSE."""
    W, Xclean = c['W_dev'], c['X_s_dev']
    if isinstance(sel_result, tuple) and sel_result[0] == 'xy':
        _, Wv, Xv = sel_result
        model, lik, sc = L.train_gp_arrays(Wv, Xv, device=DEVICE, iters=100, seed=seed)
        val = val_idx
    else:
        idx = sel_result
        model, lik, sc = L.train_gp(W, Xtrain_src, idx, device=DEVICE, iters=100, seed=seed)
        val = np.setdiff1d(val_idx, idx)
    m = L.healthy_fit_metrics(model, lik, sc, W, Xclean, val, device=DEVICE)  # 평가는 항상 clean
    return m['rmse']


def exp_stage_A(c, val_idx, fracs=(0.0, 0.1, 0.2, 0.3), k_sigma=6.0, seeds=(0, 1, 2)):
    print('\n' + '=' * 78)
    print('Stage A — 학습점 선택 robust (출력 스파이크 주입 → clean held-out RMSE)')
    print('  낮을수록 좋음. 오염↑에도 RMSE가 안 오르면 robust.')
    print('=' * 78, flush=True)
    Xclean = c['X_s_dev']
    std = Xclean.std(axis=0)
    Xwork = Xclean.copy()  # 공유 버퍼 1개만 할당
    nr = c['normal_ranges']
    healthy_rows = np.concatenate([np.arange(s, e + 1) for s, e in nr])
    methods = {
        'baseline(linspace)': select_baseline,
        'A-1(window median)':  select_A1_window_median,
        'A-2(Hampel+linsp)':   select_A2_hampel,
    }
    # table[name][p] = list over seeds
    table = {name: [[] for _ in fracs] for name in methods}
    for pi, p in enumerate(fracs):
        for sd in seeds:
            hit = spike_rows(Xwork, Xclean, healthy_rows, p, k_sigma, sd, std)
            for name, fn in methods.items():
                sel = fn(c, Xwork)
                table[name][pi].append(fit_and_eval(c, sel, val_idx, Xwork, seed=sd))
            restore_rows(Xwork, Xclean, hit)
        print(f'  [done p={p:.0%}]', flush=True)
    table = {name: [np.mean(col) for col in rows] for name, rows in table.items()}
    print('\n  ' + f"{'method':22s}" + ''.join(f'p={p:<6.0%}' for p in fracs))
    print('  ' + '-' * 74)
    for name, row in table.items():
        print(f'  {name:22s}' + ''.join(f'{v:<8.4f}' for v in row))
    print('  ' + '-' * 74)
    base = table['baseline(linspace)']
    for name, row in table.items():
        if name == 'baseline(linspace)':
            continue
        # 오염 견딤 = 가장 높은 오염에서 baseline 대비 RMSE 개선
        imp = (base[-1] - row[-1]) / base[-1] * 100
        print(f'  p={fracs[-1]:.0%}에서 {name:22s} baseline 대비 ΔRMSE={imp:+.2f}% '
              f'{"(오염에 강함)" if imp>0 else ""}')
    return table


def exp_stage_A3(c, val_idx, frac=0.2, k_sigma=6.0, seeds=(0, 1, 2)):
    """A-3 steady-state: transient(높은 |dW/dt|) 위치에 스파이크 주입 → steady 선택이 견디는가."""
    print('\n' + '=' * 78)
    print('Stage A-3 — steady-state 필터 (transient 위치 스파이크 주입)')
    print('=' * 78)
    W, Xclean, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    nr = c['normal_ranges']

    # 각 정상구간에서 |dW/dt| 계산, transient(상위 30%) vs steady(하위) 구분
    steady_pool, trans_pool = [], []
    for (s, e) in nr:
        seg = W[s:e + 1].astype(np.float64)
        dW = np.zeros(len(seg)); dW[1:] = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        thr = np.quantile(dW, 0.7)
        gidx = np.arange(s, e + 1)
        steady_pool.append(gidx[dW <= thr]); trans_pool.append(gidx[dW > thr])
    steady_pool = np.concatenate(steady_pool); trans_pool = np.concatenate(trans_pool)

    def sel_steady(K=20):
        out = []
        for (s, e) in nr:
            seg = W[s:e + 1].astype(np.float64)
            dW = np.zeros(len(seg)); dW[1:] = np.linalg.norm(np.diff(seg, axis=0), axis=1)
            gidx = np.arange(s, e + 1)
            pool = gidx[dW <= np.quantile(dW, 0.7)]
            out.append(linspace_in_pool(pool, K))
        return np.sort(np.concatenate(out)).astype(np.int64)

    base_idx = c['dev_normal_idx']
    steady_idx = sel_steady()
    std = Xclean.std(axis=0)
    Xwork = Xclean.copy()
    results = {}
    for label, idx in [('baseline(linspace)', base_idx), ('A-3(steady-state)', steady_idx)]:
        vals = []
        for sd in seeds:
            hit = spike_rows(Xwork, Xclean, trans_pool, frac, k_sigma, sd, std)  # transient에만 오염
            vals.append(fit_and_eval(c, idx, val_idx, Xwork, seed=sd))
            restore_rows(Xwork, Xclean, hit)
        ntr = np.isin(idx, trans_pool).mean() * 100
        results[label] = np.mean(vals)
        print(f'  {label:22s} 오염후 RMSE={np.mean(vals):.4f}  (선택점 중 transient비율={ntr:.0f}%)')
    imp = (results['baseline(linspace)'] - results['A-3(steady-state)']) / results['baseline(linspace)'] * 100
    print(f'  → A-3가 baseline 대비 ΔRMSE={imp:+.2f}% {"(transient 오염에 강함)" if imp>0 else ""}')


# ─────────────────────────────────────────────
# Stage B : 잔차 cycle 집계 robust
# ─────────────────────────────────────────────
def agg_mean(x):     return np.mean(x, axis=0)
def agg_median(x):   return np.median(x, axis=0)
def agg_trim(x):     return stats.trim_mean(x, 0.2, axis=0)
def agg_madrej(x):
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0) * 1.4826 + 1e-8
    out = np.empty(x.shape[1])
    for j in range(x.shape[1]):
        keep = np.abs(x[:, j] - med[j]) <= 3 * mad[j]
        out[j] = x[keep, j].mean() if keep.any() else med[j]
    return out

AGGS = {'mean(baseline)': agg_mean, 'median': agg_median,
        'trimmed20%': agg_trim, 'MAD-reject': agg_madrej}


def build_traj(res, units, cycles, agg_fn):
    """res(N,14) → dict[unit][cycle]=집계벡터(14). HI 센서만 뒤에서 추림."""
    out = {}
    for u in np.unique(units):
        um = units == u
        uc = cycles[um]; ur = res[um]
        d = {}
        for cyc in np.unique(uc):
            d[int(cyc)] = agg_fn(ur[uc == cyc])
        out[int(u)] = d
    return out


def traj_dev(traj_c, traj_clean, sensor_cols):
    """오염 trajectory vs clean trajectory의 평균 절대편차(HI 센서)."""
    devs = []
    for u in traj_clean:
        for cyc in traj_clean[u]:
            a = traj_c[u][cyc][sensor_cols]; b = traj_clean[u][cyc][sensor_cols]
            devs.append(np.abs(a - b))
    return float(np.mean(devs))


def exp_stage_B(c, fracs=(0.0, 0.1, 0.2, 0.3), k_sigma=6.0, seeds=(0, 1, 2)):
    print('\n' + '=' * 78)
    print('Stage B — 잔차 cycle 집계 robust (잔차에 이상치 주입 → clean trajectory와의 편차)')
    print('  편차 낮을수록 robust. 오염↑에도 clean 곡선을 유지하면 좋음.')
    print('=' * 78, flush=True)
    W, Xclean, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    abn_i, abn_c = c['dev_abn_indices'], c['dev_abn_cycles']
    sensor_cols = np.array([L.OUTPUT_NAMES.index(s) for s in SENSORS])

    # clean GP 1회 학습 → 깨끗한 per-point 잔차
    model, lik, sc = L.train_gp(W, Xclean, c['dev_normal_idx'], device=DEVICE, iters=100, seed=0)
    mean_pred, _ = L.gp_predict(model, lik, sc, W, abn_i, device=DEVICE)
    res_clean = Xclean[abn_i] - mean_pred
    units = A[abn_i, 0].astype(int); cyc = abn_c.astype(int)
    res_std = res_clean.std(axis=0)

    # 기준(clean) trajectory: 각 집계법별
    clean_traj = {name: build_traj(res_clean, units, cyc, fn) for name, fn in AGGS.items()}

    table = {name: [[] for _ in fracs] for name in AGGS}
    rng_master = np.random.default_rng(0)
    for pi, p in enumerate(fracs):
        for sd in seeds:
            rng = np.random.default_rng(1000 * sd + 7)
            res_c = res_clean.copy()
            n = int(len(res_c) * p)
            if n > 0:
                hit = rng.choice(len(res_c), n, replace=False)
                sgn = rng.choice([-1, 1], size=(n, res_c.shape[1]))
                res_c[hit] = res_clean[hit] + sgn * k_sigma * res_std
            for name, fn in AGGS.items():
                tr = build_traj(res_c, units, cyc, fn)
                table[name][pi].append(traj_dev(tr, clean_traj[name], sensor_cols))
        print(f'  [done p={p:.0%}]', flush=True)
    table = {name: [np.mean(col) for col in rows] for name, rows in table.items()}
    print('\n  ' + f"{'aggregator':18s}" + ''.join(f'p={p:<6.0%}' for p in fracs))
    print('  ' + '-' * 70)
    for name, row in table.items():
        print(f'  {name:18s}' + ''.join(f'{v:<8.4f}' for v in row))
    print('  ' + '-' * 70)
    base = table['mean(baseline)']
    for name, row in table.items():
        if name == 'mean(baseline)':
            continue
        imp = (base[-1] - row[-1]) / (base[-1] + 1e-12) * 100
        print(f'  p={fracs[-1]:.0%}: {name:14s} baseline(mean) 대비 편차 {imp:+.1f}% '
              f'{"(오염에 강함)" if imp>0 else ""}')
    return table


def main():
    t0 = time.time()
    c = L.load_cache()
    val_idx = L.dense_healthy_pool(c['normal_ranges'], per_range=300, seed=999)
    print(f'캐시 로드. held-out clean 검증풀={len(val_idx)}')
    exp_stage_A(c, val_idx)
    exp_stage_A3(c, val_idx)
    exp_stage_B(c)
    print(f'\n총 소요 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
