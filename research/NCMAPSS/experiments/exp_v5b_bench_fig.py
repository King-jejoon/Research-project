"""
exp_v5b_bench_fig.py — fig_bench_rul_box.png re-rendered from the tenth-
opening batch (v5b_test_batch.npz, re-frozen l12 downstream).  Style
identical to exp_v4_bench_figs.fig_rul_box; the 'mogp' column is the final
conditional-gate chain.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = __import__('exp_paths').PAPER
MODELS = ['mogp', 'llke', 'bspline', 'lr', 'cabn']
KEY = {'mogp': 'chain'}

Z = np.load(os.path.join(HERE, 'v5b_test_batch.npz'))
data = []
for n in MODELS:
    k = KEY.get(n, n)
    P, T = Z[f'{k}_P'], Z[f'{k}_T']
    ns = len(P) // 3                 # per seed: frac-major, unit-minor
    vals = []
    for iu in range(6):
        errs = [P[s * ns + j * 6 + iu] - T[s * ns + j * 6 + iu]
                for s in range(3) for j in range(4)]
        vals.append(float(np.sqrt(np.mean(np.square(errs)))))
    data.append(vals)
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.boxplot(data, tick_labels=MODELS, widths=0.55, whis=(0, 100))
ax.set_xlabel('Normal model'); ax.set_ylabel('RUL RMSE')
ax.set_title('Test RUL RMSE per unit and model, frozen downstream')
ax.grid(True, ls=':', alpha=0.4, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_bench_rul_box.png'), dpi=150)
plt.close(fig)
print('written')
