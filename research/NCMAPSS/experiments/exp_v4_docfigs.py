"""
exp_v4_docfigs.py — v4 document support: the three artefacts the v6 results
document still needs (all analysis-only on cached stats).

  A. detector sensitivity grid at the frozen gate V*=20.2 (set5 stats):
     cycle windows {3,4,5,6,8,12,20} and flight-hour windows
     {10,15,20,25,30,50} x clip {10,12,14,16} sigma
     -> table + fig_v4_sens_cycle_ext.png / fig_v4_sens_flighth_ext.png
  B. sensor screening ranking bar chart (v4_t1_screen.npz, top-5 in blue)
     -> fig_v4_sensitivity.png
  C. HI trajectories on the test units (seed 0, frozen HI recipe)
     -> fig_v4_hi_curves_test.png
Output log: v4_docfigs_results.txt
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import meandrop
from exp_v4_setcmp import load as load_stats, SEEDS
from exp_rul_r23 import HI_CFG, train_model_tail
from exp_v4_test import summaries, trim25

VSTAR = 20.2
L1, L2 = 8.0, 0.5
OUT = __import__('exp_paths').PAPER
RES = os.path.join(HERE, 'v4_docfigs_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def sens_grid():
    log('A — extended-low detector sensitivity grid (gate V*=20.2, set5)')
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
    CLIPS = [10, 12, 14, 16]
    SR = {}
    for mode, wins in [('cyc', CYC), ('fh', FH)]:
        for wdw in wins:
            for cl in CLIPS:
                e = []
                for cur, dur, ucyc, onset in curves.values():
                    if mode == 'cyc':
                        w = max(3, min(int(wdw), len(cur) - 5))
                    else:
                        w = max(3, int(np.searchsorted(np.cumsum(dur),
                                                       float(wdw)) + 1))
                    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
                    k = meandrop(np.maximum(cur, mu0 - cl * sd0))
                    det = (int(ucyc[k]) if (k is not None and k < len(ucyc))
                           else int(ucyc[-1] + 1))
                    e.append(det - onset)
                e = np.array(e, float)
                SR[(mode, wdw, cl)] = float(np.sqrt((e ** 2).mean()))
    log('  clip | ' + ' | '.join(f'{w:>3}cy' for w in CYC) + ' || '
        + ' | '.join(f'{w:>3}h' for w in FH))
    for cl in CLIPS:
        log(f'  {cl:>4} | '
            + ' | '.join(f'{SR[("cyc", w, cl)]:5.2f}' for w in CYC) + ' || '
            + ' | '.join(f'{SR[("fh", w, cl)]:5.2f}' for w in FH))
    MARK = ['o', 's', '^', 'D', 'v', 'P', 'X']
    for mode, wins, unit, title, fname in [
            ('cyc', CYC, 'cycles', 'Cycle-count baseline windows',
             'fig_v4_sens_cycle_ext.png'),
            ('fh', FH, 'flight-h', 'Flight-hour baseline windows',
             'fig_v4_sens_flighth_ext.png')]:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for i, w in enumerate(wins):
            ax.plot(CLIPS, [SR[(mode, w, cl)] for cl in CLIPS],
                    marker=MARK[i % 7], ls='--', ms=6, lw=1.4,
                    label=f'{w} {unit}')
        ax.set_xlabel('Clip strength  c'); ax.set_ylabel('Onset RMSE [cycles]')
        ax.set_title(title); ax.set_xticks(CLIPS)
        ax.grid(True, ls=':', alpha=0.5)
        ax.legend(title='Baseline window', frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)
        log(f'  figure: {fname}')
    return SR


def rank_fig():
    log('B — sensor screening ranking figure')
    Z = np.load(os.path.join(HERE, 'v4_t1_screen.npz'))
    z, order, names = Z['zrms'], Z['order'], [str(n) for n in Z['names']]
    sel = {'T30', 'T48', 'T50', 'Nc', 'Wf'}
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    xs = np.arange(len(order))
    vals = [z[j] for j in order]
    cols = ['tab:blue' if names[j] in sel else 'lightgray' for j in order]
    ax.bar(xs, vals, color=cols)
    ax.axhline(1.0, color='gray', lw=1.0, ls=':')
    ax.set_xticks(xs); ax.set_xticklabels([names[j] for j in order],
                                          fontsize=8)
    ax.set_ylabel('Degradation zRMS\n(healthy-noise units)')
    ax.set_title('Sensor sensitivity screening (blue = selected set)')
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_v4_sensitivity.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    log(f'  figure: {p}')


def hi_test_fig():
    log('C — HI trajectories on test units (seed 0)')
    sd = 0
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    d_units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                      if k.endswith('_cc')})
    dev_raw = {}
    for u in d_units:
        cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= VSTAR)
            T[i] = trim25(rs[m if m.any() else b])
        dev_raw[u] = T
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                      if k.endswith('_cc')})
    t_raw, *_ = summaries(Zt, VSTAR, t_units)
    allr = np.concatenate([dev_raw[u] for u in d_units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Zn_d = {u: (dev_raw[u] - mu) / sg for u in d_units}
    Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
    cfg = dict(HI_CFG); cfg.update(lambda1=L1, lambda2=L2, end_target=1.03)
    np.random.seed(sd)
    him, _ = train_model_tail([Zn_d[u] for u in d_units], **cfg)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in t_units:
        h = him.forward(Zn_t[u]).flatten()
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title('HI trajectories (TEST units, dev-frozen network)')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    p = os.path.join(OUT, 'fig_v4_hi_curves_test.png')
    fig.savefig(p, dpi=150); plt.close(fig)
    log(f'  figure: {p}')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    sens_grid()
    rank_fig()
    hi_test_fig()
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
