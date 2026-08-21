"""
exp_v5c_resid_fig.py — bench residual grids with the proposed healthy-range
MOGP added as the TOP row (user request 2026-08-15).  Layout identical to
exp_v4_bench_figs._grid (per-cycle mean residual, model x sensor, raw
units, seed 0); cabn dropped (predictions identical to lr).  The existing
doc figures are NOT overwritten — new filenames with a _prop suffix.
Outputs: 논문/fig_bench_dev_resid_prop.png,
         논문/fig_bench_test_resid_prop.png
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
MODELS = ['proposed', 'mogp', 'llke', 'bspline', 'lr']
CMAP = plt.get_cmap('tab10')


def stats_xyz(path):
    """(cc, uu, resid) pooled from a per-unit stats npz."""
    Z = np.load(os.path.join(HERE, path))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    cc, uu, rs = [], [], []
    for u in units:
        cc.append(Z[f'u{u}_cc']); rs.append(Z[f'u{u}_resid'])
        uu.append(np.full(len(Z[f'u{u}_cc']), u))
    return (np.concatenate(cc), np.concatenate(uu),
            np.concatenate(rs)), units


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


def fig_dev():
    Z = np.load(os.path.join(HERE, 'v4_bench_resid.npz'))
    prop, _ = stats_xyz('v5c_stats_s0.npz')
    d = {'proposed': prop,
         'mogp': (Z['mogp_cc'], Z['mogp_uu'], Z['mogp_resid'])}
    for n in ('llke', 'bspline', 'lr'):
        d[n] = (Z['cc'], Z['uu'], Z[f'{n}_resid'])
    _grid(d, Z['units'],
          'Per-cycle mean residual per model and sensor, development '
          'units, raw units, seed 0', 'fig_bench_dev_resid_prop.png')


def fig_test():
    Zt = np.load(os.path.join(HERE, 'v4_bench_test_resid_s0.npz'))
    prop, t_units = stats_xyz('v5c_test_stats_s0.npz')
    mog, _ = stats_xyz('v4_test_stats_s0.npz')
    d = {'proposed': prop, 'mogp': mog}
    for n in ('llke', 'bspline', 'lr'):
        d[n] = (Zt['cc'], Zt['uu'], Zt[f'{n}_resid'])
    _grid(d, np.array(t_units),
          'Per-cycle mean residual per model and sensor, test units, '
          'raw units, seed 0', 'fig_bench_test_resid_prop.png')


if __name__ == '__main__':
    fig_dev(); print('dev ok')
    fig_test(); print('test ok')
