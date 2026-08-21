"""
make_blockdiagram_v2.py — Figure 1 redesign: serial two-stage layout.

Front end feeds stage 1 (detection: conditional gate -> onset); the
detected healthy range feeds stage 2 (retraining -> ungated residuals ->
HI -> first passage -> RUL).  No branch after the gate.  Shaded blocks
are learned or calibrated components.  English labels, no parentheses.
Output: fig_blockdiagram_v2.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
GRAY = '#d9d9d9'


def box(ax, x, y, w, h, title, sub, shaded=False, fs=10.5, fss=8.3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.15,rounding_size=0.35',
        facecolor=GRAY if shaded else 'white', edgecolor='black', lw=1.1))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha='center', va='center',
                fontsize=fs, fontweight='bold')
        ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center',
                fontsize=fss, color='0.25')
    else:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center',
                fontsize=fs, fontweight='bold')
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='-|>', lw=1.3, color='black',
                                shrinkA=2, shrinkB=2))


def handoff(ax, xa, ya, xb, yb, ymid):
    """down from a, horizontal at ymid, arrow down into b."""
    ax.plot([xa, xa], [ya, ymid], color='black', lw=1.3)
    ax.plot([xa, xb], [ymid, ymid], color='black', lw=1.3)
    arrow(ax, xb, ymid, xb, yb)


def main():
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 31)
    ax.axis('off')

    # ---- front end (shared) ----
    fy, fh = 24.4, 5.2
    f1 = box(ax, 2.0, fy, 21, fh, 'Raw sensor data +',
             'operating conditions')
    f2 = box(ax, 27.5, fy, 21, fh, 'Sensor screening', '14 sensors to 5',
             shaded=True)
    f3 = box(ax, 53.0, fy, 21, fh, 'Stage-1 MOGP',
             'cycle 1..3, Vecchia RBF rank 1', shaded=True)
    f4 = box(ax, 78.5, fy, 19.5, fh, 'Residuals + confidence',
             'r = x − x̂,  detcov')
    for a, b in [(f1, f2), (f2, f3), (f3, f4)]:
        arrow(ax, a[0] + a[2] + 0.15, fy + fh / 2, b[0] - 0.15, fy + fh / 2)

    # ---- stage 1 band: detection ----
    ax.add_patch(Rectangle((0.8, 13.2), 98.4, 8.4, facecolor='#f2f6fa',
                           edgecolor='0.55', ls='--', lw=0.9))
    ax.text(1.8, 20.6, 'Stage 1 — detection', fontsize=10,
            fontweight='bold', color='0.30', va='center')
    sy, sh = 14.2, 5.2
    s1 = box(ax, 8.0, sy, 22, sh, 'Conditional gate',
             'keep detcov ≥ V — 20.2, sparse 19.0', shaded=True)
    s2 = box(ax, 34.5, sy, 22, sh, 'q75 likelihood curve',
             '30 flight-h baseline, clip 10σ')
    s3 = box(ax, 61.0, sy, 22, sh, 'Mean-drop change point',
             'onset cycle per unit', shaded=True)
    s4 = box(ax, 87.5, sy, 10.5, sh, 'Onset', '')
    for a, b in [(s1, s2), (s2, s3), (s3, s4)]:
        arrow(ax, a[0] + a[2] + 0.15, sy + sh / 2, b[0] - 0.15, sy + sh / 2)
    handoff(ax, f4[0] + f4[2] / 2, fy - 0.15,
            s1[0] + s1[2] / 2, sy + sh + 0.25, 22.9)

    # ---- stage 2 band: RUL ----
    ax.add_patch(Rectangle((0.8, 2.0), 98.4, 8.4, facecolor='#faf6f0',
                           edgecolor='0.55', ls='--', lw=0.9))
    ax.text(1.8, 9.4, 'Stage 2 — RUL', fontsize=10, fontweight='bold',
            color='0.30', va='center')
    ry, rh = 3.0, 5.2
    r1 = box(ax, 8.0, ry, 15.6, rh, 'Detected healthy range',
             'cycles before onset', fs=9.4, fss=7.9)
    r2 = box(ax, 26.1, ry, 15.0, rh, 'Stage-2 MOGP',
             'matched 27k rows, ungated', shaded=True, fs=9.4, fss=7.9)
    r3 = box(ax, 43.6, ry, 15.6, rh, 'trim25 + z-normalization',
             'pooled dev mean, s.d.', fs=9.4, fss=7.9)
    r4 = box(ax, 61.7, ry, 14.0, rh, 'HI network', 'MLP, shape losses',
             shaded=True, fs=9.4, fss=7.9)
    r5 = box(ax, 78.2, ry, 13.4, rh, 'First passage',
             'h = a e^bc + d = 1', shaded=True, fs=9.4, fss=7.9)
    r6 = box(ax, 94.1, ry, 4.6, rh, 'RUL', '', fs=9.8)
    for a, b in [(r1, r2), (r2, r3), (r3, r4), (r4, r5), (r5, r6)]:
        arrow(ax, a[0] + a[2] + 0.15, ry + rh / 2, b[0] - 0.15, ry + rh / 2)
    handoff(ax, s4[0] + s4[2] / 2, sy - 0.15,
            r1[0] + r1[2] / 2, ry + rh + 0.25, 11.8)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_blockdiagram_v2.png'), dpi=170,
                bbox_inches='tight')
    print('written fig_blockdiagram_v2.png')


if __name__ == '__main__':
    main()
