"""Test-engine curves under the FINAL system (causal drop 85%, L_flat HI).
fig_test_residuals.png : per-unit cycle-mean filtered residuals (3 sensors, z-scored)
fig_test_hi_v2.png        : per-unit HI trajectories (vs cycle, vs life fraction)
Seed 0 network; residuals identical across seeds up to NPER subsampling.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_test_curves.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_gp_selfdrop import build_resid, SENS, HI_KW
from neural_fusion_tail import train_model_tail

SD, DROP = 0, 0.85
SENSOR_C = {'T48': '#028090', 'T50': '#E76F51', 'Wf': '#6A4C93'}
UNIT_C = ['#028090', '#E76F51', '#6A4C93', '#2A9D8F', '#C0504D', '#5B7DB1']

z = np.load(os.path.join(HERE, f'port_stats_s3_matern32_r1_m18_n8192_p200_s{SD}.npz'))
dl, tl = build_resid(z, DROP)
np.random.seed(SD)
him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0, lambda1=2.0, lambda2=0.25, init_threshold=0.2, alpha=0.001, flat_w=300, flat_m=0.002, flat2_w=500, flat2_m=0.006)
tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
tCY = {u: cy for u, cy, d in tl}
tRD = {u: d for u, cy, d in tl}

# ---- fig 1: residuals per test unit ----
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
for ax, (u, cy, d) in zip(axes.flat, tl):
    for j, s in enumerate(SENS):
        ax.plot(cy, d[:, j], lw=1.6, color=SENSOR_C[s], label=s)
    ax.axhline(0, color='#999999', lw=0.8, ls=':')
    ax.set_title(f'test unit {u}  ({len(cy)} cycles)', fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
for ax in axes[1]: ax.set_xlabel('cycle')
for ax in axes[:, 0]: ax.set_ylabel('residual (z-scored)')
axes[0, 0].legend(fontsize=9, loc='upper left')
fig.suptitle('Test-engine residuals after the uncertainty filter '
             '(per-cycle causal detcov, drop 85%, NPER=200, seed 0)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(HERE, 'fig_test_residuals.png'), dpi=170); plt.close(fig)

# ---- fig 2: HI per test unit ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
ax = axes[0]
for k, (u, cy, d) in enumerate(tl):
    ax.plot(cy, tHI[u], lw=2, color=UNIT_C[k])
    ax.annotate(f'u{u}', xy=(cy[-1], tHI[u][-1]), fontsize=9, color=UNIT_C[k],
                xytext=(3, 0), textcoords='offset points')
ax.axhline(1.0, color='#C0504D', lw=1, ls='--')
ax.text(1, 1.015, 'failure  h = 1', fontsize=8.5, color='#C0504D')
ax.set_xlabel('cycle'); ax.set_ylabel('health index')
ax.set_title('(a) HI vs cycle'); ax.grid(alpha=0.25, lw=0.5)

ax = axes[1]
for k, (u, cy, d) in enumerate(tl):
    x = np.linspace(0, 1, len(cy))
    ax.plot(x * 100, tHI[u], lw=2, color=UNIT_C[k], label=f'unit {u}')
ax.axhline(1.0, color='#C0504D', lw=1, ls='--')
ax.set_xlabel('life fraction (%)'); ax.set_title('(b) HI vs normalized life')
ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=9, ncol=2, loc='upper left')

fig.suptitle('Test-engine health index — final HI network '
             '($\\lambda$=(1,1,2,0.25) + $L_{flat}$(300,.002) + $L_{flat2}$(500,.006), drop 85%, seed 0)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(HERE, 'fig_test_hi_v2.png'), dpi=170); plt.close(fig)

for u in sorted(tHI):
    h = tHI[u]
    print(f'unit {u}: n={len(h)}  start={h[:max(1,len(h)//10)].mean():.3f}  end={h[-1]:.3f}')
print('saved fig_test_residuals.png, fig_test_hi_v2.png')
