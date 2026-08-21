"""
exp_v4_docfigs4.py — v6 r4: figures 6-8 as RMSE-valued box plots
(user decision: keep the box-plot form, box the per-unit RMSE values;
full-range whiskers, so no separate outlier markers and no unit hidden).

  fig_v4_rul_box.png       test RUL RMSE per truncation; each box holds the
                           per-unit RMSE over 3 seeds (6 test units)
  fig_v4_sens_cycle.png    onset RMSE per cycle-count baseline window at the
                           frozen clip 10 sigma; per-unit RMSE over 3 seeds
  fig_v4_sens_flighth.png  same for flight-hour windows
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_docfigs2 import sens_errors

OUT = __import__('exp_paths').PAPER
FRACS = [0.2, 0.4, 0.6, 0.8]


def fig_rul_box():
    Z = np.load(os.path.join(HERE, 'v4_final.npz'))
    P, T, F = Z['P'], Z['T'], Z['F']
    n = len(P) // 3                       # frac-major, unit-minor per seed
    data = []
    for j, f in enumerate(FRACS):
        vals = []
        for iu in range(6):
            errs = [P[s * n + j * 6 + iu] - T[s * n + j * 6 + iu]
                    for s in range(3)]
            vals.append(float(np.sqrt(np.mean(np.square(errs)))))
        data.append(vals)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.boxplot(data, tick_labels=['20%', '40%', '60%', '80%'], widths=0.55,
               whis=(0, 100))
    ax.set_xlabel('Truncation point as share of life')
    ax.set_ylabel('RUL RMSE')
    ax.set_title('Test RUL RMSE by truncation')
    ax.grid(True, ls=':', alpha=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_rul_box.png'), dpi=150)
    plt.close(fig)


def fig_sens_boxes():
    CYC, FH, E = sens_errors()
    for mode, wins, xlab, title, fname in [
            ('cyc', CYC, 'Baseline window in cycles',
             'Cycle-count baseline windows', 'fig_v4_sens_cycle.png'),
            ('fh', FH, 'Baseline window in flight hours',
             'Flight-hour baseline windows', 'fig_v4_sens_flighth.png')]:
        data = []
        for w in wins:
            e = np.array(E[(mode, w)]).reshape(3, 9)   # [seed, unit]
            data.append(np.sqrt((e ** 2).mean(0)))
        fig, ax = plt.subplots(figsize=(5.6, 3.7))
        ax.boxplot(data, tick_labels=[str(w) for w in wins], widths=0.6,
                   whis=(0, 100))
        ax.set_xlabel(xlab); ax.set_ylabel('Onset RMSE')
        ax.set_title(title)
        ax.grid(True, ls=':', alpha=0.4, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    fig_rul_box(); print('rul box ok')
    fig_sens_boxes(); print('sens boxes ok')
