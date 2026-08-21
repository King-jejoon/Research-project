"""
추천 방법들을 순서대로 적용하고, baseline 대비 '실제' 개선이 있는지 검증.
기존 코드는 호출만 한다.

평가지표(올바른 방향):
  - held-out 정상 적합도 RMSE / NLL : 낮을수록 좋은 baseline (학습점이 healthy manifold를 잘 대표하는가)
  - 열화 SNR                         : 높을수록 좋음 (열화신호가 정상잡음 위로 뚜렷한가)
  - seed 강건성(표준편차)            : 낮을수록 좋음 (랜덤성에 불변인가)
  - 잔차 백색성 검정 통과율          : 높을수록 좋음 (baseline이 구조를 모두 흡수 → 잔차는 잡음뿐)
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)

import exp_lib as L
from compare_median_sampling import find_representative_indices  # 기존 코드 호출

SENSORS_HI = ['T48', 'T50', 'Wf']   # 노트북에서 HI에 쓰는 센서
DEVICE = 'cpu'


# ─────────────────────────────────────────────
# 학습점 선택 전략들 (budget 동일하게 맞춤)
# ─────────────────────────────────────────────
def kcenter_select(W, candidate_pool, budget, seed=0):
    """k-center greedy(farthest-point) : 표준화된 비행조건(W) 공간을 최대 커버.
    max-entropy / D-optimal space-filling 설계의 결정론적 근사."""
    rng = np.random.default_rng(seed)
    cand = np.asarray(candidate_pool)
    Xc = W[cand].astype(np.float64)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-8
    Xn = (Xc - mu) / sd
    n = len(cand)
    start = int(rng.integers(n))
    chosen = [start]
    d = np.linalg.norm(Xn - Xn[start], axis=1)
    for _ in range(budget - 1):
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(Xn - Xn[nxt], axis=1))
    return np.sort(cand[np.array(chosen)]).astype(np.int64)


def get_selectors(c):
    W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    nr = [tuple(r) for r in c['normal_ranges']]
    eng = A[:, 0].astype(int)
    base = c['dev_normal_idx']
    budget = len(base)

    # 공통 후보 풀(각 range에서 촘촘히) — k-center/representative 후보
    pool = L.dense_healthy_pool(c['normal_ranges'], per_range=400, seed=7)

    def sel_baseline(seed):
        return base  # linspace, 결정론적
    def sel_random(seed):
        return L.random_pick([tuple(r) for r in c['normal_ranges']], K=budget // len(nr), seed=seed)
    def sel_kcenter(seed):
        return kcenter_select(W, pool, budget, seed=seed)
    def sel_repidx(seed):
        # 기존 함수 호출: baseline 점들을 window 대표점으로 대체
        return find_representative_indices(W, X, base, A, normal_ranges=nr,
                                           half_win=100, top_k_init=10, top_k_step=5).astype(np.int64)

    return {
        'baseline(linspace)': sel_baseline,
        'random':             sel_random,
        'repidx(±100)':       sel_repidx,
        'kcenter(B-1)':       sel_kcenter,
    }


# ─────────────────────────────────────────────
# Exp B : 학습점 선택 → held-out 정상 적합도
# ─────────────────────────────────────────────
def exp_B(c, val_idx):
    W, X = c['W_dev'], c['X_s_dev']
    print('\n' + '=' * 74)
    print('Exp B — 학습점 선택 전략별 held-out 정상 적합도 (단일 seed=0)')
    print('=' * 74)
    sels = get_selectors(c)
    rows = {}
    for name, fn in sels.items():
        idx = fn(0)
        val = np.setdiff1d(val_idx, idx)
        model, lik, sc = L.train_gp(W, X, idx, device=DEVICE, iters=100, seed=0)
        m = L.healthy_fit_metrics(model, lik, sc, W, X, val, device=DEVICE)
        rows[name] = m
        print(f'  {name:20s}  n={len(idx):4d}  RMSE={m["rmse"]:.4f}  NLL={m["nll"]:8.3f}')
    b = rows['baseline(linspace)']
    print('  ' + '-' * 70)
    for name, m in rows.items():
        if name == 'baseline(linspace)':
            continue
        dr = (m['rmse'] - b['rmse']) / b['rmse'] * 100
        dn = m['nll'] - b['nll']
        print(f'  {name:20s}  ΔRMSE={dr:+6.2f}%  ΔNLL={dn:+7.3f}  '
              f'{"개선" if dr < 0 else "저하"}(RMSE)')
    return rows


# ─────────────────────────────────────────────
# Exp A : 잔차 정의 (mean vs z-score) → 열화 SNR
# ─────────────────────────────────────────────
def exp_A(c, val_idx):
    W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    abn_i, abn_c = c['dev_abn_indices'], c['dev_abn_cycles']
    print('\n' + '=' * 74)
    print('Exp A — 잔차 정의: 평균잔차(baseline) vs z-score잔차(A-1) → 열화 SNR')
    print('=' * 74)
    idx = c['dev_normal_idx']
    val = np.setdiff1d(val_idx, idx)
    model, lik, sc = L.train_gp(W, X, idx, device=DEVICE, iters=100, seed=0)

    for zs in [False, True]:
        # 노이즈 바닥을 '동일 정의'로 정확히 측정 (z-score면 z-score healthy std)
        hnoise, _ = L.healthy_residual_std(model, lik, sc, W, X, val, device=DEVICE, zscore=zs)
        traj = L.residual_trajectory(model, lik, sc, W, X, A, abn_i, abn_c,
                                     device=DEVICE, zscore=zs)
        snr = L.degradation_snr(traj, hnoise, SENSORS_HI)
        tag = 'z-score(A-1)' if zs else 'mean(baseline)'
        line = '  '.join(f'{s}={snr[s]:.2f}' for s in SENSORS_HI)
        nfloor = '  '.join(f'{s}={hnoise[s]:.3g}' for s in SENSORS_HI)
        print(f'  {tag:16s}  SNR  {line}  | 평균={np.nanmean(list(snr.values())):.2f}')
        print(f'  {"":16s}  노이즈바닥 {nfloor}')
    return model, lik, sc, val


# ─────────────────────────────────────────────
# Exp D : 잔차 백색성 검정 (baseline 정당성)
# ─────────────────────────────────────────────
def exp_D(c, model, lik, sc, val):
    from scipy import stats
    W, X = c['W_dev'], c['X_s_dev']
    print('\n' + '=' * 74)
    print('Exp D — 정상 잔차 백색성 검정 (잔차의 랜덤성=잡음뿐임을 증명)')
    print('  · zero-mean: 잔차 평균=0 (편향 없음)   · LjungBox p>0.05: 시간 자기상관 없음(백색)')
    print('  · |skew|,|exkurt|: 0에 가까울수록 정규    · Shapiro는 n=400 부분표본')
    print('=' * 74)
    # 시계열 백색성은 '순차 구간'에서 평가: 가장 긴 정상 구간 선택
    nr = c['normal_ranges']
    lengths = nr[:, 1] - nr[:, 0]
    s, e = nr[int(np.argmax(lengths))]
    seg_idx, seg_res = L.ordered_segment_residuals(model, lik, sc, W, X, int(s), int(e),
                                                   stride=50, device=DEVICE)
    # zero-mean/정규성은 독립 held-out 풀에서
    _, pool_res = L.healthy_residual_std(model, lik, sc, W, X, val, device=DEVICE)
    print(f'  순차구간 길이={len(seg_idx)}점(stride50),  정규성표본={len(pool_res)}점')
    n_zero = n_white = 0
    for j, name in enumerate(L.OUTPUT_NAMES):
        r = pool_res[:, j]; r = r[np.isfinite(r)]
        sr = seg_res[:, j]
        if r.std() < 1e-9:
            continue
        zero_t = stats.ttest_1samp(r, 0.0)[1]
        rs = (r - r.mean()) / r.std()
        skew = stats.skew(rs); exk = stats.kurtosis(rs)
        sh = stats.shapiro(rs[:400])[1]
        _, lb_p = L.ljung_box(sr, lags=10)
        ok_zero = zero_t > 0.05
        ok_white = lb_p > 0.05
        n_zero += int(ok_zero); n_white += int(ok_white)
        print(f'  {name:6s} mean={r.mean():+.2e} zero-mean p={zero_t:.2f}{"✓" if ok_zero else " "} '
              f'LjungBox p={lb_p:.2f}{"✓" if ok_white else " "} '
              f'|skew|={abs(skew):.2f} |exkurt|={abs(exk):.2f} Shapiro={sh:.2f}')
    print(f'  → zero-mean 통과 {n_zero}/14,  시간무상관(LjungBox) 통과 {n_white}/14')


# ─────────────────────────────────────────────
# Exp C : seed 강건성 (랜덤성에 불변인가)
# ─────────────────────────────────────────────
def exp_C(c, val_idx, seeds=(0, 1, 2, 3, 4)):
    W, X = c['W_dev'], c['X_s_dev']
    print('\n' + '=' * 74)
    print(f'Exp C — seed 강건성 ({len(seeds)} seeds): held-out RMSE 평균±std (낮은 std=강건)')
    print('=' * 74)
    sels = get_selectors(c)
    for name, fn in sels.items():
        rmses = []
        for s in seeds:
            idx = fn(s)
            val = np.setdiff1d(val_idx, idx)
            model, lik, sc = L.train_gp(W, X, idx, device=DEVICE, iters=100, seed=s)
            m = L.healthy_fit_metrics(model, lik, sc, W, X, val, device=DEVICE)
            rmses.append(m['rmse'])
        rmses = np.array(rmses)
        det = 'deterministic' if rmses.std() < 1e-9 else ''
        print(f'  {name:20s}  RMSE {rmses.mean():.4f} ± {rmses.std():.4f}  '
              f'[min {rmses.min():.4f}, max {rmses.max():.4f}] {det}')


def exp_validate_kcenter(c, val_idx, seeds=range(8)):
    """kcenter 개선이 '진짜'인지 검증: seed별 baseline vs kcenter 쌍비교 + 부호검정."""
    from scipy import stats
    W, X = c['W_dev'], c['X_s_dev']
    print('\n' + '=' * 74)
    print('검증 — kcenter(B-1) vs baseline 쌍비교 (개선이 우연/특정 seed 산물인지 확인)')
    print('=' * 74)
    base = c['dev_normal_idx']
    pool = L.dense_healthy_pool(c['normal_ranges'], per_range=400, seed=7)
    budget = len(base)
    diffs, wins = [], 0
    # baseline은 linspace(결정론적)이라 1회만; init seed만 변동
    for s in seeds:
        vb = np.setdiff1d(val_idx, base)
        mb, lb_, scb = L.train_gp(W, X, base, device=DEVICE, iters=100, seed=s)
        rb = L.healthy_fit_metrics(mb, lb_, scb, W, X, vb, device=DEVICE)['rmse']
        kc = kcenter_select(W, pool, budget, seed=s)
        vk = np.setdiff1d(val_idx, kc)
        mk, lk, sck = L.train_gp(W, X, kc, device=DEVICE, iters=100, seed=s)
        rk = L.healthy_fit_metrics(mk, lk, sck, W, X, vk, device=DEVICE)['rmse']
        d = rb - rk  # 양수면 kcenter가 더 좋음
        diffs.append(d); wins += int(d > 0)
        print(f'  seed {s}:  baseline={rb:.4f}  kcenter={rk:.4f}  Δ={d:+.4f}  '
              f'{"kcenter승" if d>0 else "baseline승"}')
    diffs = np.array(diffs)
    t_p = stats.ttest_1samp(diffs, 0.0)[1]
    print('  ' + '-' * 70)
    print(f'  평균개선 ΔRMSE = {diffs.mean():+.4f} ± {diffs.std():.4f}  '
          f'({diffs.mean()/0.91*100:+.2f}% of baseline)')
    print(f'  kcenter 승률 = {wins}/{len(diffs)}  |  paired t-test p={t_p:.4g}  '
          f'→ {"유의미한 개선(유효)" if t_p<0.05 and diffs.mean()>0 else "유의하지 않음"}')


def main():
    t0 = time.time()
    c = L.load_cache()
    val_idx = L.dense_healthy_pool(c['normal_ranges'], per_range=300, seed=999)
    print(f'데이터/캐시 로드 완료. held-out 정상 검증 풀 = {len(val_idx)}점')

    rows = exp_B(c, val_idx)
    model, lik, sc, val = exp_A(c, val_idx)
    exp_D(c, model, lik, sc, val)
    exp_C(c, val_idx, seeds=(0, 1, 2, 3, 4))
    exp_validate_kcenter(c, val_idx, seeds=range(8))
    print(f'\n총 소요 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
