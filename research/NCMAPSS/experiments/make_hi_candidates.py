"""Visual check: test HI curves for 4 candidate (mono, conv) settings.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_hi_candidates.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_hi_sweep import build_residuals
import neural_fusion as NF

SD = 0
dl, tl = build_residuals(SD)
CANDS = [
    ('candidate A:  mono=1, conv=0.5   (RMSE 7.93 / score 22.0)', 1.0, 0.5),
    ('candidate B:  mono=2, conv=0.5   (RMSE 7.95 / score 22.2)', 2.0, 0.5),
    ('candidate C:  mono=2, conv=0.25  (RMSE 7.94 / score 22.0)', 2.0, 0.25),
    ('previous:     mono=6, conv=2     (RMSE 8.25 / score 25.3)', 6.0, 2.0),
]

fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
for ax, (name, lm, lc) in zip(axes.ravel(), CANDS):
    np.random.seed(SD)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                            lambda1=lm, lambda2=lc, init_threshold=0.2,
                            alpha=0.001, verbose=False)
    for u, cy, d in tl:
        h = him.forward(d).flatten()
        ax.plot(np.array(cy) / max(cy), h, lw=1.6, alpha=0.85, label=f'unit {u}')
    ax.axhline(1.0, color='r', ls='--', lw=1)
    ax.axhline(0.2, color='gray', ls=':', lw=1)
    ax.set_ylim(0, 1.15); ax.grid(alpha=0.3)
    ax.set_title(name, fontsize=10.5)
for ax in axes[1]: ax.set_xlabel('life fraction')
for ax in axes[:, 0]: ax.set_ylabel('Health Index')
axes[0, 0].legend(fontsize=8, ncol=2, loc='upper left')
fig.suptitle('HI shape check — candidate loss weights (test units, seed 0, drop 60%)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, 'hi_candidates.png')
fig.savefig(out, dpi=160)
print('saved', out)
