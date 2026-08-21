"""
exp_v5_docfigs.py — final-report figures re-rendered on the coverage-
conditional chain (user decision 2026-08-10).  Same visual conventions as
the fixed-gate versions (exp_v4_docfigs2/4): full-range whiskers, per-unit
RMSE boxes, seed-0 curves.

  fig_v5_sens_cycle.png      onset RMSE per cycle-count window, conditional
  fig_v5_sens_flighth.png    same for flight-hour windows
  fig_v5_hi_curves.png       dev HI, conditional tables, seed 0
  fig_v5_hi_curves_test.png  test HI, conditional tables, seed 0
  fig_v5_rul_box.png         test RUL RMSE per truncation, conditional chain
                             (v5_full_rul.npz — the ninth-opening summary)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import HI_CFG, train_model_tail
from exp_v4_final import med3
from exp_v5_full_rul import dev_tables, test_tables

OUT = __import__('exp_paths').PAPER
L1, L2 = 12.0, 0.25                    # re-frozen recipe (2026-08-10)
CYC = [3, 4, 5, 6, 8, 12, 20]
FH = [10, 15, 20, 25, 30, 50]


def fig_sens_boxes():
    Z = np.load(os.path.join(HERE, 'v5_sens.npz'))
    for mode, wins, xlab, title, fname in [
            ('cyc', CYC, 'Baseline window in cycles',
             'Cycle-count baseline windows', 'fig_v5_sens_cycle.png'),
            ('fh', FH, 'Baseline window in flight hours',
             'Flight-hour baseline windows', 'fig_v5_sens_flighth.png')]:
        data = []
        for w in wins:
            e = Z[f'{mode}_{w}_e'].reshape(3, 9)          # [seed, unit]
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


def fig_hi():
    sd = 0
    units, raw, Zn, hrs, mu, sg = dev_tables(sd)
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    np.random.seed(sd)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in units:
        h = med3(him.forward(Zn[u]).flatten())
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title('HI trajectories of development units')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v5_hi_curves.png'), dpi=150)
    plt.close(fig)

    t_units, t_raw = test_tables(sd)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in t_units:
        h = med3(him.forward((t_raw[u] - mu) / sg).flatten())
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title('HI trajectories of test units')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v5_hi_curves_test.png'), dpi=150)
    plt.close(fig)


def fig_rul_box():
    Z = np.load(os.path.join(HERE, 'v5b_test_batch.npz'))
    P, T = Z['chain_P'], Z['chain_T']
    FRACS = [0.2, 0.4, 0.6, 0.8]
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
    fig.savefig(os.path.join(OUT, 'fig_v5_rul_box.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_sens_boxes(); print('sens boxes ok')
    fig_hi(); print('hi curves ok')
    fig_rul_box(); print('rul box ok')
