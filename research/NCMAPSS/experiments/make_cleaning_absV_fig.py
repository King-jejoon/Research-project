"""Table 5 replacement figure: cleaning on the ABSOLUTE detcov axis.
x = absolute detcov threshold V (keep detcov >= V), y = healthy-gen zRMSE
with seed s.d. error bars.  V is the independent variable (colleague style,
`detcov > 3.60`); removed fraction is annotated only as a by-product."""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = __import__('exp_paths').PAPER
z = np.load(os.path.join(HERE, 'cleaning_absV_sweep.npz'))
V, zg, zsd, zi, rem = (z['V_grid'], z['zgn_mean'], z['zgn_sd'],
                       z['zin_mean'], z['removed_mean'])
m = ~np.isnan(V)                       # drop the no-filter row from the curve
base = float(zg[~m][0])                # no-filter baseline zRMSE

fig, ax = plt.subplots(figsize=(6.0, 4.0))
ax.errorbar(V[m], zg[m], yerr=zsd[m], marker='o', ls='--', color='tab:blue',
            ms=7, lw=1.6, capsize=3, label='healthy-gen zRMSE')
ax.plot(V[m], zi[m], marker='s', ls=':', color='tab:red', ms=6, lw=1.3,
        label='held-in guard (coverage)')
ax.axhline(base, color='gray', ls='-', lw=1.0)
ax.text(V[m][0], base + 0.012, f'no filter: {base:.3f}',
        fontsize=8, color='gray', va='bottom')

# annotate the removed fraction on the top axis
axt = ax.twiny()
axt.set_xlim(ax.get_xlim())
axt.set_xticks(V[m])
axt.set_xticklabels([f'{r:.0%}' for r in rem[m]], fontsize=8)
axt.set_xlabel('removed fraction (derived, not a control)', fontsize=9)

ax.set_xlabel('absolute detcov threshold  V   (keep points with detcov >= V)')
ax.set_ylabel('zRMSE')
ax.grid(True, ls=':', alpha=0.5)
ax.legend(frameon=False, loc='upper left')
fig.tight_layout()
p = os.path.join(OUT, 'fig_cleaning_absV.png')
fig.savefig(p, dpi=150); plt.close(fig)
print('written', p)
print('overlap check: min zRMSE %.3f+-%.3f  vs  V=20.00 %.3f+-%.3f'
      % (zg[m].min(), zsd[m][int(np.argmin(zg[m]))], zg[m][-3], zsd[m][-3]))
