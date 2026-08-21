"""
exp_detcov_gpquality.py — colleague-style detcov(불확실성) 검사를 내 DS03
파이프라인 통계에 접목. 평가 기준 = GP 예측 품질 (실측 vs 예측 잔차),
RUL/score 미사용.

검사 방식 (colleague):  X -> 조건부 공분산 행렬식 (detcov = -0.5*logdet S)
                        detcov가 낮은 점(불확실성 큰 극단 조건) 제거
평가 (user 지시):       필터 전/후 GP가 실제 센서값을 얼마나 잘 맞추는가
  - healthy(고장 전) 사이클 잔차 RMSE  : 낮을수록 정상모델 적합 좋음
  - degraded(고장 후) 잔차 RMS         : 열화 신호 보존 확인
  - detcov ~ |residual| 상관           : 불확실성이 실제 오차를 예측하는가
  - separation = (deg RMS - healthy RMS) / healthy RMS

입력: port_stats_s3_rbf_r2_m18_n8192_p200_lr0.1_s{0,1,2}.npz
      (동결 파이프라인 GP: 3센서 T48/T50/Wf, RBF rank2, NPER=200, mcond=15)
"""
import os
import numpy as np
from scipy import stats as sstats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = 's3_rbf_r2_m18_n8192_p200_lr0.1'
SEEDS = [0, 1, 2]
SENS = ['T48', 'T50', 'Wf']
DROPS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
RES = os.path.join(HERE, 'detcov_gpquality_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def true_onset(hs, ucyc):
    below = np.where(hs < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def load_seed(sd):
    z = np.load(os.path.join(HERE, f'port_stats_{TAG}_s{sd}.npz'))
    units = sorted({k.rsplit('_', 1)[0] for k in z.files})
    out = {}
    for u in units:
        d = dict(cc=z[f'{u}_cc'], resid=z[f'{u}_resid'], detcov=z[f'{u}_detcov'])
        if f'{u}_hs' in z.files:
            ucyc = np.unique(d['cc'])
            d['onset'] = true_onset(z[f'{u}_hs'], ucyc)
            d['healthy'] = d['cc'] < d['onset']
        out[u] = d
    return out


def main():
    open(RES, 'w').close()
    log(f'DETCOV-BASED INSPECTION ON MY DS03 PIPELINE STATS  tag={TAG}')
    log(f'  evaluation = GP prediction quality (residual vs actual), no RUL/score')
    log('')

    per_seed_tables = []
    for sd in SEEDS:
        S = load_seed(sd)
        devu = [u for u in S if u.startswith('dev')]
        tstu = [u for u in S if u.startswith('test')]

        dc_dev = np.concatenate([S[u]['detcov'] for u in devu])
        r_dev = np.concatenate([S[u]['resid'] for u in devu])
        h_dev = np.concatenate([S[u]['healthy'] for u in devu])
        dc_tst = np.concatenate([S[u]['detcov'] for u in tstu])
        r_tst = np.concatenate([S[u]['resid'] for u in tstu])

        # -- detcov ~ |resid| correlation on healthy dev points --
        rn = np.linalg.norm(r_dev[h_dev] / r_dev[h_dev].std(0), axis=1)
        rho, pv = sstats.spearmanr(dc_dev[h_dev], rn)
        log(f'seed{sd}: healthy dev pts={h_dev.sum()}  degraded={len(h_dev)-h_dev.sum()}  '
            f'spearman(detcov, |resid|) = {rho:.3f} (p={pv:.1e})')

        rows = []
        for dr in DROPS:
            thr = np.percentile(dc_dev, dr * 100) if dr > 0 else -np.inf
            kd = dc_dev >= thr
            kt = dc_tst >= thr
            # healthy / degraded RMSE per sensor (raw units)
            mh = kd & h_dev
            md = kd & ~h_dev
            rmse_h = np.sqrt((r_dev[mh] ** 2).mean(0))
            rms_d = np.sqrt((r_dev[md] ** 2).mean(0))
            rmse_t = np.sqrt((r_tst[kt] ** 2).mean(0))
            # z-scored overall (comparable across sensors)
            zstd = r_dev[h_dev].std(0)
            zh = float(np.sqrt(((r_dev[mh] / zstd) ** 2).mean()))
            zd = float(np.sqrt(((r_dev[md] / zstd) ** 2).mean()))
            zt = float(np.sqrt(((r_tst[kt] / zstd) ** 2).mean()))
            sep = (zd - zh) / zh
            rows.append(dict(dr=dr, thr=thr, keep=float(kd.mean()), zh=zh, zd=zd,
                             zt=zt, sep=sep, rmse_h=rmse_h, rms_d=rms_d, rmse_t=rmse_t))
        per_seed_tables.append(rows)

    log('')
    log(f'{"drop%":>6} | {"kept%":>6} | {"healthy zRMSE":>13} | {"degraded zRMS":>13} | '
        f'{"test zRMSE":>10} | {"separation":>10}')
    log('-' * 78)
    for j, dr in enumerate(DROPS):
        zh = [t[j]['zh'] for t in per_seed_tables]
        zd = [t[j]['zd'] for t in per_seed_tables]
        zt = [t[j]['zt'] for t in per_seed_tables]
        sp = [t[j]['sep'] for t in per_seed_tables]
        kp = np.mean([t[j]['keep'] for t in per_seed_tables])
        log(f'{dr:>6.0%} | {kp:>6.1%} | {np.mean(zh):>7.4f}±{np.std(zh):>5.4f} | '
            f'{np.mean(zd):>7.4f}±{np.std(zd):>5.4f} | {np.mean(zt):>10.4f} | '
            f'{np.mean(sp):>10.3f}')

    log('')
    log('per-sensor healthy RMSE (raw units, seed mean), drop=0% vs 10% vs 30%:')
    for j, dr in enumerate(DROPS):
        if dr not in (0.0, 0.10, 0.30):
            continue
        rh = np.mean([t[j]['rmse_h'] for t in per_seed_tables], axis=0)
        rd = np.mean([t[j]['rms_d'] for t in per_seed_tables], axis=0)
        log(f'  drop={dr:>4.0%}: healthy ' +
            '  '.join(f'{s}={v:.3f}' for s, v in zip(SENS, rh)) +
            ' | degraded ' + '  '.join(f'{s}={v:.3f}' for s, v in zip(SENS, rd)))

    # ---------- figures (seed 0) ----------
    S = load_seed(0)
    devu = [u for u in S if u.startswith('dev')]
    dc = np.concatenate([S[u]['detcov'] for u in devu])
    r = np.concatenate([S[u]['resid'] for u in devu])
    h = np.concatenate([S[u]['healthy'] for u in devu])
    zstd = r[h].std(0)
    rn = np.linalg.norm(r / zstd, axis=1)

    fig = plt.figure(figsize=(13, 4))
    ax = plt.subplot(1, 3, 1)
    bins = np.linspace(dc.min(), dc.max(), 60)
    ax.hist(dc[h], bins=bins, alpha=0.6, label='healthy', density=True)
    ax.hist(dc[~h], bins=bins, alpha=0.6, label='degraded', density=True)
    ax.set_xlabel('detcov  (high = low uncertainty)')
    ax.set_title('detcov distribution (dev, seed0)')
    ax.legend()

    ax = plt.subplot(1, 3, 2)
    qs = np.percentile(dc[h], np.linspace(0, 100, 11))
    ce = [rn[h][(dc[h] >= qs[i]) & (dc[h] < qs[i + 1] if i < 9 else dc[h] <= qs[i + 1])].mean()
          for i in range(10)]
    ax.plot(np.arange(10) * 10 + 5, ce, 'o-')
    ax.set_xlabel('detcov decile (low->high)')
    ax.set_ylabel('mean |z-resid| (healthy)')
    ax.set_title('uncertainty vs actual GP error')

    ax = plt.subplot(1, 3, 3)
    for j, dr in enumerate(DROPS):
        pass
    zh0 = [np.mean([t[j]['zh'] for t in per_seed_tables]) for j in range(len(DROPS))]
    zd0 = [np.mean([t[j]['zd'] for t in per_seed_tables]) for j in range(len(DROPS))]
    ax.plot([d * 100 for d in DROPS], zh0, 'o-', label='healthy zRMSE')
    ax.plot([d * 100 for d in DROPS], zd0, 's-', label='degraded zRMS')
    ax.set_xlabel('dropped lowest-detcov %')
    ax.set_title('GP fit quality vs filter strength')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, 'fig_detcov_gpquality.png'), dpi=130)
    plt.close(fig)

    # example unit curves: per-cycle mean |z-resid| raw vs 10% filter
    fig = plt.figure(figsize=(13, 6))
    thr10 = np.percentile(dc, 10)
    for k, u in enumerate(devu[:6]):
        ax = plt.subplot(2, 3, k + 1)
        d = S[u]
        ucyc = np.unique(d['cc'])
        raw = [np.linalg.norm(d['resid'][d['cc'] == c] / zstd, axis=1).mean() for c in ucyc]
        mkk = d['detcov'] >= thr10
        fil = [np.linalg.norm(d['resid'][(d['cc'] == c) & mkk] / zstd, axis=1).mean()
               if ((d['cc'] == c) & mkk).any() else np.nan for c in ucyc]
        ax.plot(ucyc, raw, '.', color='gray', label='raw')
        ax.plot(ucyc, fil, color='darkblue', label='detcov filter 10%')
        ax.axvline(d['onset'], color='red', lw=1.5)
        ax.set_title(f'{u} (onset={d["onset"]})', fontsize=9)
        if k == 0:
            ax.legend(fontsize=7)
    plt.suptitle('DS03 dev units: per-cycle mean |z-resid|, raw vs detcov-filtered')
    plt.subplots_adjust(wspace=0.25, hspace=0.45)
    plt.savefig(os.path.join(HERE, 'fig_detcov_gpquality_units.png'), dpi=130)
    plt.close(fig)
    log('')
    log('figures: fig_detcov_gpquality.png, fig_detcov_gpquality_units.png')


if __name__ == '__main__':
    main()
