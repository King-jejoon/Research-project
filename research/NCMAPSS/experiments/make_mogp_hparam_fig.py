"""Paper figure: MOGP hyperparameter validation.
(a) GPR learning-rate sweep (RUL RMSE, 3 seeds)  (b) sensor set x kernel grid.
Values from port_*_results.txt (3-seed mean +/- std).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_mogp_hparam_fig.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
TEAL, GRAY, RED = '#028090', '#808080', '#C0504D'

# ---- (a) lr sweep: s3/m32r1, NPER=200, drop 60%, 120 steps fixed ----
lrs = np.array([0.01, 0.02, 0.05, 0.10])
rmse = np.array([12.58, 11.75, 8.25, 9.31])
rmse_sd = np.array([2.09, 1.57, 0.12, 0.40])

# ---- (b) sensor set x kernel grid: NPER=60, no filter ----
grid = np.array([[8.81, 8.85],    # 3-sensor:  matern32 r1 | rbf r2
                 [9.20, 9.55]])   # 5-sensor
grid_sd = np.array([[0.40, 0.34], [0.30, 0.09]])
rows = ['3 sensors\n(T48, T50, Wf)', '5 sensors\n(+T30, Nc)']
cols = ['Matern 3/2\nrank 1', 'RBF\nrank 2']

teal_cmap = LinearSegmentedColormap.from_list('teal', ['#FFFFFF', TEAL])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

ax = axes[0]
ax.errorbar(lrs, rmse, yerr=rmse_sd, color=TEAL, lw=2, marker='o', ms=6, capsize=3)
ax.scatter([0.05], [8.25], s=170, facecolors='none', edgecolors=RED, lw=2,
           zorder=5, label='selected: lr = 0.05')
ax.set_xscale('log')
ax.set_xticks(lrs); ax.set_xticklabels([str(v) for v in lrs])
ax.set_xlabel('GPR learning rate (Adam, 120 steps)')
ax.set_ylabel('RUL RMSE (cycles)')
ax.set_title('(a) Learning-rate sweep')
ax.grid(alpha=0.25, lw=0.6)
ax.legend(frameon=True, loc='upper right')
ax.annotate('non-converged', xy=(0.014, 12.2), fontsize=9, color=GRAY)

ax = axes[1]
im = ax.imshow(grid, cmap=teal_cmap, vmin=8.5, vmax=9.8, alpha=0.85)
for i in range(2):
    for j in range(2):
        ax.text(j, i, f'{grid[i, j]:.2f}\n± {grid_sd[i, j]:.2f}', ha='center',
                va='center', fontsize=11,
                color='white' if grid[i, j] > 9.3 else '#1a1a1a')
ax.add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor=RED, lw=2.5))
ax.set_xticks([0, 1]); ax.set_xticklabels(cols)
ax.set_yticks([0, 1]); ax.set_yticklabels(rows)
ax.set_title('(b) Sensor set x kernel (RUL RMSE)')
plt.colorbar(im, ax=ax, fraction=0.046, label='RMSE (cycles)')

fig.suptitle('MOGP hyperparameter validation (3 seeds, mean ± std)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, 'fig_mogp_hparam.png')
fig.savefig(out, dpi=180)
print('saved', out)
