"""
exp_v4_docfigs3.py — v6 r3 figures: grid heatmap replacing Table 2, and
per-family onset-error boxplots replacing the line sensitivity figures
(no stars).  Labels contain no parentheses or brackets.
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

GRID = {
    'Top-3 set': {
        'RBF':        [(0.01668, 0.00229), (0.03523, 0.02805), (0.03389, 0.02610)],
        'Matern 3/2': [(0.01738, 0.00166), (0.01739, 0.00165), (0.01733, 0.00165)],
        'Matern 1/2': [(0.01787, 0.00284), (0.01799, 0.00292), (0.01812, 0.00310)]},
    'Top-5 set': {
        'RBF':        [(0.01223, 0.00085), (0.01341, 0.00193), (0.01345, 0.00195)],
        'Matern 3/2': [(0.01364, 0.00141), (0.01387, 0.00143), (0.01393, 0.00145)],
        'Matern 1/2': [(0.01563, 0.00346), (0.01575, 0.00353), (0.01594, 0.00369)]},
    'Top-7 set': {
        'RBF':        [(0.01038, 0.00071), (0.01042, 0.00070), (0.01052, 0.00073)],
        'Matern 3/2': [(0.01170, 0.00115), (0.01160, 0.00133), (0.01173, 0.00126)],
        'Matern 1/2': [(0.01412, 0.00328), (0.01392, 0.00326), (0.01406, 0.00326)]},
}


def fig_heatmap():
    """design A4: Blues, one shared absolute colorbar on the right,
    scale capped at 0.020 (values above saturate to the lightest tone)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap('Blues')
    vmin, vmax = 0.010, 0.020

    def luminance(rgb):
        r, g, b = rgb[:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.3))
    for ax, (title, K) in zip(axes, GRID.items()):
        M = np.array([[v[0] for v in K[k]] for k in K])
        val = 1 - np.clip((M - vmin) / (vmax - vmin), 0, 1)
        ax.pcolormesh(val[::-1], cmap='Blues', vmin=0, vmax=1,
                      edgecolors='white', linewidth=2)
        for i, k in enumerate(K):
            for j in range(3):
                m, s = K[k][j]
                c = cmap(val[i, j])
                txt = 'white' if luminance(c) < 0.5 else 'black'
                ax.text(j + 0.5, 2 - i + 0.5, f'{m:.4f}\n±{s:.4f}',
                        ha='center', va='center', fontsize=8.5, color=txt)
        ax.add_patch(plt.Rectangle((0.03, 2.03), 0.94, 0.94, fill=False,
                                   edgecolor='black', lw=2.2))
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels(['rank 1', 'rank 2', 'rank 3'], fontsize=9)
        ax.set_yticks([2.5, 1.5, 0.5])
        ax.set_yticklabels(list(K), fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(0, 3); ax.set_ylim(0, 3); ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.subplots_adjust(right=0.90, wspace=0.45)
    cax = fig.add_axes([0.925, 0.16, 0.016, 0.68])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax),
                                     cmap='Blues_r'), cax=cax)
    cb.ax.invert_yaxis()               # dark = low zRMSE at the top
    cb.set_ticks([0.010, 0.012, 0.014, 0.016, 0.018, 0.020])
    cb.set_ticklabels(['0.010', '0.012', '0.014', '0.016', '0.018',
                       '≥0.020'])
    cb.ax.tick_params(labelsize=8)
    cb.set_label('held-out zRMSE', fontsize=9)
    fig.savefig(os.path.join(OUT, 'fig_v4_grid_heatmap.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def fig_sens_boxes():
    CYC, FH, E = sens_errors()
    for mode, wins, xlab, title, fname in [
            ('cyc', CYC, 'Baseline window in cycles',
             'Cycle-count baseline windows', 'fig_v4_sens_cycle.png'),
            ('fh', FH, 'Baseline window in flight hours',
             'Flight-hour baseline windows', 'fig_v4_sens_flighth.png')]:
        fig, ax = plt.subplots(figsize=(5.6, 3.7))
        ax.boxplot([E[(mode, w)] for w in wins],
                   tick_labels=[str(w) for w in wins], widths=0.6)
        ax.axhline(0, color='gray', lw=1.0, ls=':')
        ax.set_xlabel(xlab); ax.set_ylabel('Onset error')
        ax.set_title(title)
        ax.grid(True, ls=':', alpha=0.4, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)


if __name__ == '__main__':
    fig_heatmap(); print('heatmap ok')
    fig_sens_boxes(); print('sens boxes ok')
