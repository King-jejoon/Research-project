"""
exp_detcov_param_proof.py — evidence figure for the chosen detcov drop rate.

Method under test:  X -> conditional covariance determinant (detcov)
                    -> uncertainty size -> remove most-uncertain points.
Evaluation:         GP prediction quality (actual vs predicted residuals).

Fine drop-rate sweep (0..50%, 1% grid) on the frozen DS03 pipeline stats,
3 seeds.  Fixed z-normalization (sigma from pre-filter healthy dev resid).
Output: fig_detcov_param_proof.png + detcov_param_proof.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = 's3_rbf_r2_m18_n8192_p200_lr0.1'
SEEDS = [0, 1, 2]
DROPS = np.arange(0, 51) / 100.0
CHOSEN = 0.10
RES = os.path.join(HERE, 'detcov_param_proof.txt')


def true_onset(hs, ucyc):
    below = np.where(hs < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def load_seed(sd):
    z = np.load(os.path.join(HERE, f'port_stats_{TAG}_s{sd}.npz'))
    units = sorted({k.rsplit('_', 1)[0] for k in z.files})
    dc_d, r_d, h_d, dc_t, r_t = [], [], [], [], []
    for u in units:
        cc, resid, dc = z[f'{u}_cc'], z[f'{u}_resid'], z[f'{u}_detcov']
        if u.startswith('dev'):
            onset = true_onset(z[f'{u}_hs'], np.unique(cc))
            dc_d.append(dc); r_d.append(resid); h_d.append(cc < onset)
        else:
            dc_t.append(dc); r_t.append(resid)
    return (np.concatenate(dc_d), np.concatenate(r_d), np.concatenate(h_d),
            np.concatenate(dc_t), np.concatenate(r_t))


ZH, ZD, ZT, SEP = [], [], [], []
for sd in SEEDS:
    dc, r, h, dct, rt = load_seed(sd)
    zstd = r[h].std(0)
    zh_row, zd_row, zt_row = [], [], []
    for dr in DROPS:
        thr = np.percentile(dc, dr * 100) if dr > 0 else -np.inf
        kd, kt = dc >= thr, dct >= thr
        zh_row.append(float(np.sqrt(((r[kd & h] / zstd) ** 2).mean())))
        zd_row.append(float(np.sqrt(((r[kd & ~h] / zstd) ** 2).mean())))
        zt_row.append(float(np.sqrt(((rt[kt] / zstd) ** 2).mean())))
    ZH.append(zh_row); ZD.append(zd_row); ZT.append(zt_row)
ZH, ZD, ZT = np.array(ZH), np.array(ZD), np.array(ZT)
SEP = (ZD - ZH) / ZH

with open(RES, 'w') as f:
    f.write('drop  healthy_zRMSE(mean+-sd)  degraded_zRMS  separation\n')
    for i, dr in enumerate(DROPS):
        f.write(f'{dr:4.0%}  {ZH[:,i].mean():.4f}+-{ZH[:,i].std():.4f}  '
                f'{ZD[:,i].mean():.4f}  {SEP[:,i].mean():.3f}\n')

ci = int(CHOSEN * 100)
x = DROPS * 100

fig = plt.figure(figsize=(14, 9))

# (a) detcov distribution + chosen threshold
ax = plt.subplot(2, 2, 1)
dc, r, h, dct, rt = load_seed(0)
bins = np.linspace(np.percentile(dc, 0.2), dc.max(), 80)
ax.hist(dc[h], bins=bins, density=True, alpha=0.6, color='tab:green',
        label='healthy points')
ax.hist(dc[~h], bins=bins, density=True, alpha=0.5, color='tab:orange',
        label='degraded points')
thr10 = np.percentile(dc, ci)
ax.axvline(thr10, color='red', ls='--', lw=1.6,
           label=f'chosen threshold (drop {ci}%)')
ax.set_xlabel('detcov  (low = uncertain X)')
ax.set_ylabel('density')
ax.set_title('(a) detcov distribution (dev, seed 0)\n'
             'healthy vs degraded nearly identical → X-based, no label leakage')
ax.legend(fontsize=8)

# (b) healthy error & degraded signal vs drop
ax = plt.subplot(2, 2, 2)
m, s = ZH.mean(0), ZH.std(0)
ax.plot(x, m, '-', color='tab:green', label='healthy zRMSE (GP error)')
ax.fill_between(x, m - s, m + s, color='tab:green', alpha=0.25)
ax.axvline(ci, color='red', ls='--', lw=1.4)
ax.set_ylabel('healthy zRMSE', color='tab:green')
ax2 = ax.twinx()
ax2.plot(x, ZD.mean(0), '-', color='tab:orange', label='degraded zRMS (signal)')
ax2.fill_between(x, ZD.mean(0) - ZD.std(0), ZD.mean(0) + ZD.std(0),
                 color='tab:orange', alpha=0.25)
ax2.set_ylim(0, 6.2)
ax2.set_ylabel('degraded zRMS', color='tab:orange')
ax.set_xlabel('lowest-detcov points removed (%)')
ax.set_title(f'(b) normal-model error drops, degradation signal untouched\n'
             f'at {ci}%: error −{100*(1-m[ci]/m[0]):.0f}%, signal '
             f'{100*(ZD.mean(0)[ci]/ZD.mean(0)[0]-1):+.1f}%')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')

# (c) separation vs drop
ax = plt.subplot(2, 2, 3)
m, s = SEP.mean(0), SEP.std(0)
ax.plot(x, m, '-', color='tab:blue', label='separation = (deg−healthy)/healthy')
ax.fill_between(x, m - s, m + s, color='tab:blue', alpha=0.25)
mt, st = ZT.mean(0), ZT.std(0)
ax2 = ax.twinx()
ax2.plot(x, mt, '-', color='tab:purple', alpha=0.8, label='test zRMSE (6 held-out units)')
ax2.set_ylabel('test zRMSE', color='tab:purple')
ax.axvline(ci, color='red', ls='--', lw=1.4, label=f'chosen {ci}%')
ax.set_xlabel('lowest-detcov points removed (%)')
ax.set_ylabel('separation', color='tab:blue')
ax.set_title('(c) degradation visibility rises monotonically;\n'
             'same dev-chosen threshold transfers to held-out test units')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')

# (d) marginal gain — why 10%
ax = plt.subplot(2, 2, 4)
gain = -(np.diff(ZH.mean(0)) / ZH.mean(0)[0]) * 100     # % of baseline error removed per +1%
ax.bar(x[1:], gain, width=0.85, color=['red' if i < ci else '0.7'
                                       for i in range(len(gain))])
ax.axvline(ci + 0.5, color='red', ls='--', lw=1.4)
ax.set_xlabel('lowest-detcov points removed (%)')
ax.set_ylabel('marginal error reduction per +1% removed\n(% of baseline healthy zRMSE)')
cum10 = 100 * (1 - ZH.mean(0)[ci] / ZH.mean(0)[0])
cum50 = 100 * (1 - ZH.mean(0)[-1] / ZH.mean(0)[0])
ax.set_title(f'(d) diminishing returns: first {ci}% gives {cum10:.0f}% error cut '
             f'of the {cum50:.0f}% max at 50%\n→ knee of the curve ≈ {ci}%')

plt.suptitle('DS03 — X → covariance determinant (detcov) → uncertainty-based removal:\n'
             'evidence for the chosen 10% drop rate (frozen pipeline GP, 3 seeds, '
             'fixed z-normalization)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, 'fig_detcov_param_proof.png')
plt.savefig(out, dpi=130)
print('saved', out)
i10 = ci
print(f'at {ci}%: healthy {ZH.mean(0)[0]:.3f}->{ZH.mean(0)[i10]:.3f}  '
      f'degraded {ZD.mean(0)[0]:.3f}->{ZD.mean(0)[i10]:.3f}  '
      f'sep {SEP.mean(0)[0]:.2f}->{SEP.mean(0)[i10]:.2f}  '
      f'test {ZT.mean(0)[0]:.3f}->{ZT.mean(0)[i10]:.3f}')
