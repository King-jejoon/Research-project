"""Final verification figure: (a) test HI curves of the FINAL config
(mono=6, conv=2), (b) decile slope profiles — final vs rejected (mono=2,
conv=0.5) showing the end-flattening the rejected config suffers from.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_final_hi_fig.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_hi_sweep import build_residuals
from exp_hi_slope_check import decile_slopes
import neural_fusion as NF

SEEDS = [0, 1, 2]
FINAL = ('final (mono=6, conv=2)', 6.0, 2.0, '#028090')
REJ = ('rejected (mono=2, conv=0.5)', 2.0, 0.5, '#C0504D')

RD = {sd: build_residuals(sd) for sd in SEEDS}

def hi_curves(lm, lc, sd):
    dl, tl = RD[sd]
    np.random.seed(sd)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                            lambda1=lm, lambda2=lc, init_threshold=0.2,
                            alpha=0.001, verbose=False)
    return [(u, np.array(cy) / max(cy), him.forward(d).flatten()) for u, cy, d in tl]

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))

# (a) final-config HI curves (seed 0)
ax = axes[0]
for u, x, h in hi_curves(FINAL[1], FINAL[2], 0):
    ax.plot(x, h, lw=1.7, alpha=0.85, label=f'unit {u}')
ax.axhline(1.0, color='r', ls='--', lw=1)
ax.axhline(0.2, color='gray', ls=':', lw=1)
ax.set_ylim(0, 1.15); ax.grid(alpha=0.3)
ax.set_xlabel('life fraction'); ax.set_ylabel('Health Index')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.set_title('(a) FINAL config HI curves — test units, seed 0\n'
             'low start · convex · accelerating to 1', fontsize=10.5)

# (b) decile slope profiles (mean over units & seeds)
ax = axes[1]
xdec = np.arange(1, 11) * 10
for name, lm, lc, color in [FINAL, REJ]:
    allsl = []
    for sd in SEEDS:
        for u, x, h in hi_curves(lm, lc, sd):
            allsl.append(decile_slopes(h))
    S = np.nanmean(np.stack(allsl), 0)
    ax.plot(xdec, S, marker='o', ms=5, lw=2, color=color, label=name)
    ax.annotate(f'{S[9]:.2f}', (100, S[9]), textcoords='offset points',
                xytext=(8, -4), fontsize=9, color=color)
ax.axvspan(85, 100, color='#C0504D', alpha=0.07)
ax.text(92.5, 0.25, 'end-flattening\ncheck zone', ha='center', fontsize=8.5, color='#C0504D')
ax.set_xlabel('life-fraction window (%)'); ax.set_ylabel('HI slope in window')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper left')
ax.set_title('(b) decile slope profile — slope must keep rising\n'
             'final keeps accelerating; rejected bends down at the end', fontsize=10.5)

fig.suptitle('HI shape verification — final coefficients (1, 1, 6, 2), drop 60%', fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.91])
out = os.path.join(HERE, 'fig_hi_final_verify.png')
fig.savefig(out, dpi=170)
print('saved', out)
