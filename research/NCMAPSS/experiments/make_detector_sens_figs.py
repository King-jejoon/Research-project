"""Two detector-sensitivity figures (paper final section):
A) cycle-count windows {8,12,20}, B) flight-hour windows {30,50,80};
each vs clip strength, one line per window size.  English labels, marker+dashed."""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = __import__('exp_paths').PAPER
MARK = ['o', 's', '^']
COL = ['tab:blue', 'tab:orange', 'tab:green']


def panel(mode, unit, title, fname):
    z = np.load(os.path.join(HERE, f'detector_sens_{mode}.npz'))
    wins, clips, rmse = z['windows'], z['clips'], z['rmse']
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for i, w in enumerate(wins):
        ax.plot(clips, rmse[i], marker=MARK[i], ls='--', color=COL[i],
                ms=7, lw=1.6, label=f'{w} {unit}')
    ax.set_xlabel('Clip strength  c  (mu0 - c*sigma0)')
    ax.set_ylabel('Onset RMSE [cycles]')
    ax.set_title(title)
    ax.set_xticks(clips)
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(title='Baseline window', frameon=False)
    fig.tight_layout()
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150); plt.close(fig)
    print('written', p, '  range', rmse.min().round(2), rmse.max().round(2))


panel('cyc', 'cycles', 'Cycle-count baseline windows', 'fig_detector_sens_cycle.png')
panel('fh', 'flight-h', 'Flight-hour baseline windows', 'fig_detector_sens_flighth.png')
