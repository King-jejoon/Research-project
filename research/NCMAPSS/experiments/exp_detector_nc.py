"""
exp_detector_nc.py — STAGE 3: detector re-calibration on the NEW chain
(kNN-cleaned model, NO detcov gate — dropped by the Stage-2 ablation).

Expanded grid exactly as requested:
  windows : cycle-count {8, 12, 20}  AND  flight-hours {30, 50, 80}
  clip    : {10, 12, 14, 16} sigma
  LL quantile q : {70, 75, 80, 85, 90}   (cycle summary, ungated)
Detector: baseline window -> clip mu0 - c*sigma0 -> full-range mean-drop.
Score: onset RMSE over 9 dev units x 3 seeds (truth = hs onset).
Outputs the full grid, the dev-selected setting, and the two paper figures
(one per window family, at the selected q).
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_gate_ablation_nc import load_det, SEEDS
from exp_resid_clean import meandrop

CYC_WINS = [8, 12, 20]
FH_WINS = [30, 50, 80]
CLIPS = [10, 12, 14, 16]
QS = [70, 75, 80, 85, 90]
OUT = __import__('exp_paths').PAPER
RES = os.path.join(HERE, 'detector_nc_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def curves(D, q):
    """per (seed, unit): ungated q-quantile LL curve."""
    C = {}
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                cur[i] = np.percentile(d['ll'][d['cc'] == c], q)
            C[(sd, u)] = (cur, d['dur'], d['ucyc'], d['onset'])
    return C


def detect(cur, dur, ucyc, window, clip, mode):
    if mode == 'cyc':
        w = max(5, min(int(window), len(cur) - 5))
    else:
        w = max(5, int(np.searchsorted(np.cumsum(dur), float(window)) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - clip * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    D = {sd: load_det(sd) for sd in SEEDS}
    log('STAGE 3 — detector calibration, NEW chain (no gate), expanded grid')
    log(f'  windows {CYC_WINS} cyc + {FH_WINS} flight-h | clips {CLIPS} | q {QS}')

    R = {}
    for q in QS:
        C = curves(D, q)
        for mode, wins in [('cyc', CYC_WINS), ('fh', FH_WINS)]:
            for wdw in wins:
                for cl in CLIPS:
                    e = np.array([detect(*C[k][:3], wdw, cl, mode) - C[k][3]
                                  for k in C], float)
                    R[(q, mode, wdw, cl)] = (float(np.sqrt((e ** 2).mean())),
                                             float(e.mean()))

    for q in QS:
        log('')
        log(f'  q{q}:   clip | ' +
            ' | '.join(f'{w:>3}cyc' for w in CYC_WINS) + ' || ' +
            ' | '.join(f'{w:>3}h' for w in FH_WINS))
        for cl in CLIPS:
            row = [f'{R[(q, "cyc", w, cl)][0]:6.2f}' for w in CYC_WINS]
            row2 = [f'{R[(q, "fh", w, cl)][0]:6.2f}' for w in FH_WINS]
            log(f'        {cl:>4} | ' + ' | '.join(row) + ' || ' + ' | '.join(row2))

    best = min(R, key=lambda k: R[k][0])
    q_, mode_, w_, c_ = best
    log('')
    log(f'BEST: q{q_}, window={w_}{"cyc" if mode_=="cyc" else "flight-h"}, '
        f'clip={c_}sigma  onset RMSE={R[best][0]:.2f}  delay={R[best][1]:+.2f}')
    ref = R[(75, 'fh', 30, 10)]
    log(f'reference (old-chain setting q75/30h/10s): RMSE={ref[0]:.2f}  '
        f'delay={ref[1]:+.2f}')

    # ---- figures at the selected q ----
    MARK = ['o', 's', '^']; COL = ['tab:blue', 'tab:orange', 'tab:green']
    for mode, wins, unit, title, fname in [
            ('cyc', CYC_WINS, 'cycles', 'Cycle-count baseline windows',
             'fig_nc_detector_cycle.png'),
            ('fh', FH_WINS, 'flight-h', 'Flight-hour baseline windows',
             'fig_nc_detector_flighth.png')]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for i, w in enumerate(wins):
            ax.plot(CLIPS, [R[(q_, mode, w, cl)][0] for cl in CLIPS],
                    marker=MARK[i], ls='--', color=COL[i], ms=7, lw=1.6,
                    label=f'{w} {unit}')
        ax.set_xlabel('Clip strength  c  (mu0 - c*sigma0)')
        ax.set_ylabel('Onset RMSE [cycles]')
        ax.set_title(f'{title}  (LL q{q_}, cleaned model)')
        ax.set_xticks(CLIPS); ax.grid(True, ls=':', alpha=0.5)
        ax.legend(title='Baseline window', frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)
        log(f'figure written: {fname}')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
