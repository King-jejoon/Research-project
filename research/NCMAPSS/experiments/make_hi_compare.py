"""Compare test HI curves: base physics weights (mono=6, conv=2) vs relaxed
optimum (mono=2, conv=0.5).  Checks that relaxing constraints did not break
the HI shape (monotone-ish, convex-ish, start low, end near 1).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_hi_compare.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_hi_sweep import build_residuals, SEEDS
import neural_fusion as NF

SD = 0
dl, tl = build_residuals(SD)
configs = [('base  (mono=6, conv=2)', 6.0, 2.0), ('relaxed  (mono=2, conv=0.5)', 2.0, 0.5)]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
for ax, (name, lm, lc) in zip(axes, configs):
    np.random.seed(SD)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                            lambda1=lm, lambda2=lc, init_threshold=0.2,
                            alpha=0.001, verbose=False)
    mono_viol = []
    for u, cy, d in tl:
        h = him.forward(d).flatten()
        ax.plot(np.array(cy) / max(cy), h, lw=1.6, alpha=0.85, label=f'unit {u}')
        mono_viol.append((np.diff(h) < -1e-6).mean())
    ax.axhline(1.0, color='r', ls='--', lw=1)
    ax.axhline(0.2, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 1.15); ax.grid(alpha=0.3)
    ax.set_xlabel('life fraction')
    ax.set_title(f'{name}\nmean monotonicity violation = {np.mean(mono_viol):.1%}', fontsize=10.5)
axes[0].set_ylabel('Health Index')
axes[0].legend(fontsize=8, ncol=2, loc='upper left')
fig.suptitle('Test HI curves — physics-weight relaxation check (seed 0, drop 60%)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(HERE, 'hi_compare_relaxed.png')
fig.savefig(out, dpi=160)
print('saved', out)
