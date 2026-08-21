"""
exp_v4_bench_figs.py — figures for the benchmark report (reproducible).

  fig_bench_dev_resid_raw.png   per-cycle mean residual, model x sensor, dev units, raw units
  fig_bench_test_resid_raw.png  same on the test units
  fig_bench_rul_box.png         test RUL RMSE per unit and model

Box plots use full-range whiskers (whis=(0,100)) so no separate outlier markers
are drawn and no unit is hidden: with 6 units per box the 1.5-IQR rule flags
points spuriously, and the extreme unit (u12, the longest-lived test unit) is
common to every model.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = __import__('exp_paths').PAPER
SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
UNITS_LBL = ['K', 'K', 'K', 'rpm', 'pps']
MODELS = ['mogp', 'llke', 'bspline', 'lr', 'cabn']
CMAP = plt.get_cmap('tab10')


def _grid(cc_uu_rs, units, title, fname):
    fig, axes = plt.subplots(5, 5, figsize=(17.5, 15.5), sharex=True)
    for i, name in enumerate(MODELS):
        cc, uu, rs = cc_uu_rs[name]
        for j, s in enumerate(SENS):
            ax = axes[i, j]
            for k, u in enumerate(units):
                m = uu == u
                ucyc = np.unique(cc[m])
                cur = np.array([rs[m & (cc == c), j].mean() for c in ucyc])
                ax.plot(ucyc / ucyc.max(), cur, lw=1.0, color=CMAP(k % 10),
                        alpha=0.8)
            ax.axhline(0, color='black', lw=0.8, ls=':')
            ax.grid(True, ls=':', alpha=0.35)
            if i == 0:
                ax.set_title(s, fontsize=12)
                ax.text(0.02, 0.93, UNITS_LBL[j], transform=ax.transAxes,
                        fontsize=9, color='0.4')
            if j == 0:
                ax.set_ylabel(f'{name}\nresidual', fontsize=11)
            if i == 4:
                ax.set_xlabel('Normalized life', fontsize=10)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


def fig_dev_resid():
    Z = np.load(os.path.join(HERE, 'v4_bench_resid.npz'))
    d = {n: (Z['mogp_cc'], Z['mogp_uu'], Z['mogp_resid']) if n == 'mogp'
         else (Z['cc'], Z['uu'], Z[f'{n}_resid']) for n in MODELS}
    _grid(d, Z['units'],
          'Per-cycle mean residual per model and sensor, development units, '
          'raw units, seed 0', 'fig_bench_dev_resid_raw.png')


def fig_test_resid():
    Zt = np.load(os.path.join(HERE, 'v4_bench_test_resid_s0.npz'))
    Zm = np.load(os.path.join(HERE, 'v4_test_stats_s0.npz'))
    t_units = sorted({int(k[1:].split('_')[0]) for k in Zm.files
                      if k.endswith('_cc')})
    cc, uu, rs = [], [], []
    for u in t_units:
        cc.append(Zm[f'u{u}_cc']); rs.append(Zm[f'u{u}_resid'])
        uu.append(np.full(len(Zm[f'u{u}_cc']), u))
    mog = (np.concatenate(cc), np.concatenate(uu), np.concatenate(rs))
    d = {n: mog if n == 'mogp' else (Zt['cc'], Zt['uu'], Zt[f'{n}_resid'])
         for n in MODELS}
    _grid(d, np.array(t_units),
          'Per-cycle mean residual per model and sensor, test units, '
          'raw units, seed 0', 'fig_bench_test_resid_raw.png')


def fig_rul_box():
    Z = np.load(os.path.join(HERE, 'v4_bench_rul.npz'))
    data = []
    for n in MODELS:
        P, T = Z[f'{n}_P'], Z[f'{n}_T']
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


if __name__ == '__main__':
    fig_dev_resid(); print('dev resid ok')
    fig_test_resid(); print('test resid ok')
    fig_rul_box(); print('rul box ok')
