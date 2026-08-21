"""
exp_v4_docfigs2.py — v6 document figures, revised per user feedback:
no parentheses/brackets in axis labels or titles; revised HI recipe
(lambda1=8, lambda2=0.25, median-3 filter).

  1. fig_v4_sensitivity.png      ranking bars, y=zRMSE, no dotted line
  2. fig_v4_gate_sweep.png       star ONLY at the selected minimum V*=20.2
  3. fig_v4_hi_curves.png        dev HI, revised recipe, clean title
  4. fig_v4_hi_curves_test.png   test HI, revised recipe, clean title
  5. fig_v4_sens_cycle.png       clip 10 fixed, x = cycle window
  6. fig_v4_sens_flighth.png     clip 10 fixed, x = flight-hour window
  7. fig_v4_sens_box.png         onset-error boxplots per window, clip 10
  8. fig_v4_rul_box.png          test RUL error boxplots per truncation
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import meandrop
from exp_v4_setcmp import load as load_stats, errors, SEEDS, detect
from exp_rul_r23 import HI_CFG, train_model_tail
from exp_v4_hi import build_tables, trim25
from exp_v4_test import summaries
from exp_v4_final import med3

VSTAR = 20.2
L1, L2 = 8.0, 0.25
OUT = __import__('exp_paths').PAPER


def fig_rank():
    Z = np.load(os.path.join(HERE, 'v4_t1_screen.npz'))
    z, order, names = Z['zrms'], Z['order'], [str(n) for n in Z['names']]
    sel = {'T30', 'T48', 'T50', 'Nc', 'Wf'}
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    xs = np.arange(len(order))
    ax.bar(xs, [z[j] for j in order],
           color=['tab:blue' if names[j] in sel else 'lightgray'
                  for j in order])
    ax.set_xticks(xs); ax.set_xticklabels([names[j] for j in order],
                                          fontsize=8)
    ax.set_xlabel('Sensor'); ax.set_ylabel('zRMS')
    ax.set_title('Sensor sensitivity ranking')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_sensitivity.png'), dpi=150)
    plt.close(fig)


def fig_gate():
    D = {sd: load_stats(5, sd) for sd in SEEDS}
    e_off = errors(D, None)
    r_off = np.sqrt((e_off ** 2).mean())
    grid = np.round(np.arange(18.1, 20.5 + 1e-9, 0.1), 1)
    rows = [(float(V), float(np.sqrt((errors(D, float(V)) ** 2).mean())))
            for V in grid]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(r_off, color='gray', lw=1.2, label='No gate')
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker='o', ms=3.5,
            ls='--', color='tab:blue', label='Gated', zorder=2)
    vb, rb = min(rows, key=lambda r: r[1])
    ax.plot(vb, rb, marker='*', ms=16, color='tab:red', zorder=3,
            label='Selected V')
    ax.set_xlabel('Absolute detcov threshold V')
    ax.set_ylabel('Onset RMSE')
    ax.set_title('Inspection-gate sweep')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(frameon=False,
                                                loc='upper left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_gate_sweep.png'), dpi=150)
    plt.close(fig)


def _hi_nets(sd):
    units, Zn, hrs = build_tables(sd, VSTAR)
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    np.random.seed(sd)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    return units, Zn, him


def fig_hi_dev():
    units, Zn, him = _hi_nets(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in units:
        h = med3(him.forward(Zn[u]).flatten())
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title('HI trajectories of development units')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_hi_curves.png'), dpi=150)
    plt.close(fig)


def fig_hi_test():
    sd = 0
    units, Zn, him = _hi_nets(sd)
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    dev_raw = {}
    for u in units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c; m = b & (dc >= VSTAR)
            T[i] = trim25(rs[m if m.any() else b])
        dev_raw[u] = T
    allr = np.concatenate(list(dev_raw.values()))
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                      if k.endswith('_cc')})
    t_raw, *_ = summaries(Zt, VSTAR, t_units)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in t_units:
        h = med3(him.forward((t_raw[u] - mu) / sg).flatten())
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title('HI trajectories of test units')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_hi_curves_test.png'), dpi=150)
    plt.close(fig)


def sens_errors():
    """onset errors per window at clip 10 (27 dev cases each)."""
    D5 = {sd: load_stats(5, sd) for sd in SEEDS}
    curves = {}
    for sd in SEEDS:
        for u, d in D5[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b & (d['dc'] >= VSTAR)
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            curves[(sd, u)] = (cur, d['dur'], d['ucyc'], d['onset'])
    CYC = [3, 4, 5, 6, 8, 12, 20]; FH = [10, 15, 20, 25, 30, 50]
    E = {}
    for mode, wins in [('cyc', CYC), ('fh', FH)]:
        for wdw in wins:
            e = []
            for cur, dur, ucyc, onset in curves.values():
                if mode == 'cyc':
                    w = max(3, min(int(wdw), len(cur) - 5))
                else:
                    w = max(3, int(np.searchsorted(np.cumsum(dur),
                                                   float(wdw)) + 1))
                mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
                k = meandrop(np.maximum(cur, mu0 - 10 * sd0))
                det = (int(ucyc[k]) if (k is not None and k < len(ucyc))
                       else int(ucyc[-1] + 1))
                e.append(det - onset)
            E[(mode, wdw)] = np.array(e, float)
    return CYC, FH, E


def fig_sens(CYC, FH, E):
    for mode, wins, xlab, fname in [
            ('cyc', CYC, 'Baseline window in cycles', 'fig_v4_sens_cycle.png'),
            ('fh', FH, 'Baseline window in flight hours',
             'fig_v4_sens_flighth.png')]:
        r = [float(np.sqrt((E[(mode, w)] ** 2).mean())) for w in wins]
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.plot(wins, r, marker='o', ms=6, ls='--', color='tab:blue')
        wb = wins[int(np.argmin(r))]
        ax.plot(wb, min(r), marker='*', ms=15, color='tab:red')
        ax.set_xlabel(xlab); ax.set_ylabel('Onset RMSE')
        ax.set_title('Cycle-count baseline windows' if mode == 'cyc'
                     else 'Flight-hour baseline windows')
        ax.set_xticks(wins); ax.grid(True, ls=':', alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)


def fig_sens_box(CYC, FH, E):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax, mode, wins, xlab in [
            (axes[0], 'cyc', CYC, 'Baseline window in cycles'),
            (axes[1], 'fh', FH, 'Baseline window in flight hours')]:
        ax.boxplot([E[(mode, w)] for w in wins], tick_labels=[str(w) for w in wins],
                   showfliers=True, widths=0.6)
        ax.axhline(0, color='gray', lw=1.0, ls=':')
        ax.set_xlabel(xlab); ax.set_ylabel('Onset error')
        ax.set_title('Cycle-count windows' if mode == 'cyc'
                     else 'Flight-hour windows')
        ax.grid(True, ls=':', alpha=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_sens_box.png'), dpi=150)
    plt.close(fig)


def fig_rul_box():
    Z = np.load(os.path.join(HERE, 'v4_final.npz'))
    P, T, F = Z['P'], Z['T'], Z['F']
    fracs = [0.2, 0.4, 0.6, 0.8]
    data = [(P[np.isclose(F, f)] - T[np.isclose(F, f)]) for f in fracs]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.boxplot(data, tick_labels=['20%', '40%', '60%', '80%'], widths=0.55)
    ax.axhline(0, color='gray', lw=1.0, ls=':')
    ax.set_xlabel('Truncation point as share of life')
    ax.set_ylabel('RUL error')
    ax.set_title('Test RUL error by truncation')
    ax.grid(True, ls=':', alpha=0.4, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_v4_rul_box.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    fig_rank(); print('1 rank ok')
    fig_gate(); print('2 gate ok')
    fig_hi_dev(); print('3 hi dev ok')
    fig_hi_test(); print('4 hi test ok')
    CYC, FH, E = sens_errors()
    fig_sens(CYC, FH, E); print('5-6 sens ok')
    fig_sens_box(CYC, FH, E); print('7 sens box ok')
    fig_rul_box(); print('8 rul box ok')
