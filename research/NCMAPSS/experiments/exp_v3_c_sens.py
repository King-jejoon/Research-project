"""
exp_v3_c_sens.py — STAGE C: detector sensitivity analysis (paper final
section).  Winner set (SET env), gate fixed at V* from Stage B (VP env, pooled
percentile), summary q75.

Grid: cycle windows {8,12,20} and flight-hour windows {30,50,80}, each crossed
with clip {10,12,14,16} sigma.  Two figures, one per window family.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v3_a2cmp import load, SEEDS
from exp_resid_clean import meandrop

SET = int(os.environ.get('SET', 5))
VP = int(os.environ.get('VP', 25))
CYC_WINS = [8, 12, 20]; FH_WINS = [30, 50, 80]; CLIPS = [10, 12, 14, 16]
OUT = __import__('exp_paths').PAPER
RES = os.path.join(HERE, 'v3_c_sens_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    D = {sd: load(SET, sd) for sd in SEEDS}
    V = {sd: float(np.percentile(np.concatenate(
        [D[sd][u]['dc'] for u in D[sd]]), VP)) for sd in SEEDS}
    log(f'STAGE C — detector sensitivity, set{SET}, gate V@p{VP}, q75')

    curves = {}
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b & (d['dc'] >= V[sd])
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            curves[(sd, u)] = (cur, d['dur'], d['ucyc'], d['onset'])

    def rmse(window, clip, mode):
        e = []
        for cur, dur, ucyc, onset in curves.values():
            if mode == 'cyc':
                w = max(5, min(int(window), len(cur) - 5))
            else:
                w = max(5, int(np.searchsorted(np.cumsum(dur), float(window)) + 1))
            mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
            k = meandrop(np.maximum(cur, mu0 - clip * sd0))
            det = int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)
            e.append(det - onset)
        e = np.array(e, float)
        return float(np.sqrt((e ** 2).mean()))

    R = {}
    for mode, wins in [('cyc', CYC_WINS), ('fh', FH_WINS)]:
        for wdw in wins:
            for cl in CLIPS:
                R[(mode, wdw, cl)] = rmse(wdw, cl, mode)

    log('')
    log(f'  clip | ' + ' | '.join(f'{w:>3}cyc' for w in CYC_WINS) + ' || ' +
        ' | '.join(f'{w:>3}h' for w in FH_WINS))
    for cl in CLIPS:
        log(f'  {cl:>4} | ' +
            ' | '.join(f'{R[("cyc", w, cl)]:6.2f}' for w in CYC_WINS) + ' || ' +
            ' | '.join(f'{R[("fh", w, cl)]:6.2f}' for w in FH_WINS))
    best = min(R, key=R.get)
    log(f'  best: {best[1]}{"cyc" if best[0]=="cyc" else "flight-h"} '
        f'clip {best[2]}s -> RMSE {R[best]:.2f}')

    MARK = ['o', 's', '^']; COL = ['tab:blue', 'tab:orange', 'tab:green']
    for mode, wins, unit, title, fname in [
            ('cyc', CYC_WINS, 'cycles', 'Cycle-count baseline windows',
             'fig_v3_sens_cycle.png'),
            ('fh', FH_WINS, 'flight-h', 'Flight-hour baseline windows',
             'fig_v3_sens_flighth.png')]:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for i, w in enumerate(wins):
            ax.plot(CLIPS, [R[(mode, w, cl)] for cl in CLIPS], marker=MARK[i],
                    ls='--', color=COL[i], ms=7, lw=1.6, label=f'{w} {unit}')
        ax.set_xlabel('Clip strength  c  (mu0 - c*sigma0)')
        ax.set_ylabel('Onset RMSE [cycles]')
        ax.set_title(title)
        ax.set_xticks(CLIPS); ax.grid(True, ls=':', alpha=0.5)
        ax.legend(title='Baseline window', frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)
        log(f'  figure: {fname}')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
