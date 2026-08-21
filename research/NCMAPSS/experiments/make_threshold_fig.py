"""Paper figure: uncertainty-filter threshold (drop ratio) sweep.
Left: RUL RMSE vs drop  |  Right: NASA score vs drop.
Solid = NPER=200 (filter works), dashed gray = NPER=60 (insufficient sampling).
Values from port_s3_matern32_r1_m18_n8192[_p200]_drop*_ewma_results.txt (3 seeds).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_threshold_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# NPER=200 sweep (3-seed mean, std)
drop200 = np.array([0, 10, 15, 25, 35, 50, 60, 70, 80])
rmse200 = np.array([8.73, 8.61, 8.56, 8.50, 8.38, 8.31, 8.25, 8.23, 8.29])
rmse200_sd = np.array([0.09, 0.16, 0.18, 0.16, 0.19, 0.13, 0.12, 0.17, 0.18])
score200 = np.array([28.6, 27.3, 27.0, 26.6, 26.0, 25.6, 25.3, 25.4, 26.0])
score200_sd = np.array([0.7, 1.0, 1.0, 1.0, 1.1, 0.7, 0.6, 0.9, 0.9])

# NPER=60 sweep (insufficient per-cycle sampling -> filter hurts)
drop60 = np.array([0, 10, 15])
rmse60 = np.array([9.20, 9.26, 9.37])
rmse60_sd = np.array([0.30, 0.30, 0.29])
score60 = np.array([32.1, 32.1, 32.7])
score60_sd = np.array([2.5, 2.3, 2.4])

TEAL, GRAY, RED = '#028090', '#808080', '#C0504D'
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

for ax, (y2, s2, y6, s6, lab, best) in zip(axes, [
        (rmse200, rmse200_sd, rmse60, rmse60_sd, 'RUL RMSE (cycles)', 8.25),
        (score200, score200_sd, score60, score60_sd, "NASA PHM'08 score", 25.3)]):
    ax.errorbar(drop200, y2, yerr=s2, color=TEAL, lw=2, marker='o', ms=5,
                capsize=3, label='NPER = 200 (sufficient sampling)')
    ax.errorbar(drop60, y6, yerr=s6, color=GRAY, lw=1.6, ls='--', marker='s', ms=5,
                capsize=3, label='NPER = 60 (insufficient)')
    ax.axvspan(50, 70, color=TEAL, alpha=0.08, label='optimal plateau (50-70%)')
    ax.scatter([60], [best], s=140, facecolors='none', edgecolors=RED, lw=2,
               zorder=5, label='selected: drop 60%')
    ax.set_xlabel('drop ratio (% of points removed by detcov percentile)')
    ax.set_ylabel(lab)
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8.5, loc='upper right')
axes[0].set_title('(a) RUL RMSE vs filter threshold', fontsize=11)
axes[1].set_title('(b) NASA score vs filter threshold', fontsize=11)
fig.suptitle('Uncertainty filter threshold sweep — 3-sensor, Matern 3/2, 3 seeds (mean ± std)',
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, 'fig_threshold_sweep.png')
fig.savefig(out, dpi=200)
print('saved', out)
