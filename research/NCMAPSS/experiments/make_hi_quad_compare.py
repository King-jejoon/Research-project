"""Visual comparison of L_flat/L_flat2 candidates — test HI, seed 0.
Top row: full life. Bottom row: zoom 60-100% (where flattening hides).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_hi_quad_compare.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_gp_selfdrop import build_resid
from neural_fusion_tail import train_model_tail

SD, DROP = 0, 0.85
UNIT_C = ['#028090', '#E76F51', '#6A4C93', '#2A9D8F', '#C0504D', '#5B7DB1']
CANDS = [
    ('M: (300,.002)+f2(500,.006)\nRMSE 8.69 / 28.1', dict(flat_w=300, flat_m=0.002, flat2_w=500, flat2_m=0.006)),
    ('A: current (300,.002)\nRMSE 8.05 / 23.5', dict(flat_w=300, flat_m=0.002)),
    ('F: (300,.004)+f2(300,.004)\nRMSE 8.41 / 26.2', dict(flat_w=300, flat_m=0.004, flat2_w=300, flat2_m=0.004)),
    ('I: (300,.002)+f2(800,.004)\nRMSE 8.71 / 28.3', dict(flat_w=300, flat_m=0.002, flat2_w=800, flat2_m=0.004)),
    ('J: (300,.002)+f2(800,.006)\nRMSE 8.94 / 30.0', dict(flat_w=300, flat_m=0.002, flat2_w=800, flat2_m=0.006)),
]

z = np.load(os.path.join(HERE, f'port_stats_s3_matern32_r1_m18_n8192_p200_s{SD}.npz'))
dl, tl = build_resid(z, DROP)

fig, axes = plt.subplots(2, len(CANDS), figsize=(4.0 * len(CANDS), 7.2))
for c, (name, kw) in enumerate(CANDS):
    np.random.seed(SD)
    him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                              lambda1=2.0, lambda2=0.25, init_threshold=0.2,
                              alpha=0.001, **kw)
    for k, (u, cy, d) in enumerate(tl):
        h = him.forward(d).flatten()
        x = np.linspace(0, 1, len(h)) * 100
        for r, (lo, hi) in enumerate([(0, 100), (60, 100)]):
            ax = axes[r, c]
            m = (x >= lo)
            ax.plot(x[m], h[m], lw=1.8, color=UNIT_C[k],
                    label=f'u{u}' if r == 0 else None)
    for r, (lo, hi) in enumerate([(0, 100), (60, 100)]):
        ax = axes[r, c]
        ax.axhline(1.0, color='#C0504D', lw=1, ls='--')
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(lo, hi)
        if r == 0:
            ax.set_title(name, fontsize=10)
            ax.set_ylim(0.1, 1.1)
        else:
            ax.set_ylim(0.45, 1.1)
            ax.set_xlabel('life fraction (%)')
    axes[0, c].tick_params(labelbottom=False)
axes[0, 0].set_ylabel('health index'); axes[1, 0].set_ylabel('health index (zoom 60-100%)')
axes[0, 0].legend(fontsize=8, ncol=2, loc='upper left')
fig.suptitle('Tail-shape candidates — test HI, seed 0, drop 85%', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(HERE, 'fig_hi_quad_compare.png'), dpi=150)
print('saved fig_hi_quad_compare.png')
