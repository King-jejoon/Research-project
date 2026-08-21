"""
exp_v5c_report_figs.py — pending report renders for the two-stage
architecture (no new metrics; visualization of already-disclosed data).

  fig_hi1_curves.png / fig_hi1_curves_test.png   HI #1 (ungated cycle<=3
      tables), dev / test trajectories, seed 0
  fig_hi2_curves.png / fig_hi2_curves_test.png   HI #2 (stage-2 healthy-
      range tables, ungated), dev / test trajectories, seed 0
  fig_bench_rul_box_prop.png                     per-unit test RUL RMSE,
      Proposed / MOGP / LLKE / B-spline / LR=CaBN (frozen downstream)

Full-range whiskers on the box plot (whis=(0,100)) as in the original
bench figure, so no unit is hidden.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v5c_dev import tables as hi2_tables, L1, L2
from exp_v5c_test import test_tables as hi2_test_tables
from exp_v4_hi import trim25
from exp_rul_r23 import HI_CFG, train_model_tail
from exp_v4_final import med3

OUT = __import__('exp_paths').PAPER
SD = 0
CMAP = plt.get_cmap('tab10')


def cfg12():
    c = dict(HI_CFG); c.update(lambda1=L1, lambda2=L2, end_target=1.03)
    return c


def hi1_model():
    """HI #1: ungated cycle<=3 tables, seed 0 (dev + test curves)."""
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{SD}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{SD}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    raw = {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']
        ur = H[f'dev_{u}_ucyc']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        raw[u] = np.stack([trim25(rs[cc == c]) for c in ucyc])
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Zn = {u: (raw[u] - mu) / sg for u in units}
    np.random.seed(SD)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg12())
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{SD}.npz'))
    t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                      if k.endswith('_cc')})
    HI_t = {}
    for u in t_units:
        cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']
        ucyc = Zt[f'u{u}_ucyc']
        keep = np.isin(ucyc, np.unique(cc))
        ucyc = ucyc[keep]
        T = np.stack([trim25(rs[cc == c]) for c in ucyc])
        HI_t[u] = med3(him.forward((T - mu) / sg).flatten())
    return HIs, HI_t


def hi2_model():
    """HI #2: stage-2 healthy-range tables, ungated, seed 0."""
    units, raw, Zn, hrs, mu, sg = hi2_tables(SD)
    np.random.seed(SD)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg12())
    HIs = {u: med3(him.forward(Zn[u]).flatten()) for u in units}
    t_units, t_raw = hi2_test_tables(SD)
    HI_t = {u: med3(him.forward((t_raw[u] - mu) / sg).flatten())
            for u in t_units}
    return HIs, HI_t


def plot_hi(HIs, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, u in enumerate(sorted(HIs)):
        ax.plot(np.arange(len(HIs[u])), HIs[u], lw=1.4,
                color=CMAP(k % 10), label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.4)
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def fig_rul_box():
    C = np.load(os.path.join(HERE, 'v5c_test.npz'))
    B = np.load(os.path.join(HERE, 'v5b_test_batch.npz'))
    series = [('Proposed', C['P'], C['T']),
              ('MOGP', B['ungated_P'], B['ungated_T']),
              ('LLKE', B['llke_P'], B['llke_T']),
              ('B-spline', B['bspline_P'], B['bspline_T']),
              ('LR = CaBN', B['lr_P'], B['lr_T'])]
    data, labels = [], []
    for name, P, T in series:
        ns = len(P) // 3                     # per seed: frac-major, unit-minor
        vals = []
        for iu in range(6):
            errs = [P[s * ns + j * 6 + iu] - T[s * ns + j * 6 + iu]
                    for s in range(3) for j in range(4)]
            vals.append(float(np.sqrt(np.mean(np.square(errs)))))
        data.append(vals); labels.append(name)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.boxplot(data, tick_labels=labels, widths=0.55, whis=(0, 100))
    ax.set_xlabel('Normal model'); ax.set_ylabel('RUL RMSE')
    ax.set_title('Test RUL RMSE per unit and model, frozen downstream')
    ax.grid(True, ls=':', alpha=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_bench_rul_box_prop.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    HIs, HI_t = hi1_model()
    plot_hi(HIs, 'HI trajectories of development units', 'fig_hi1_curves.png')
    plot_hi(HI_t, 'HI trajectories of test units', 'fig_hi1_curves_test.png')
    print('hi1 ok')
    HIs, HI_t = hi2_model()
    plot_hi(HIs, 'HI trajectories of development units', 'fig_hi2_curves.png')
    plot_hi(HI_t, 'HI trajectories of test units', 'fig_hi2_curves_test.png')
    print('hi2 ok')
    fig_rul_box()
    print('box ok')
