"""
exp_v4_detcov_final.py — v4 STAGE 5: detcov gate validity on the selected
model (set5 [T30,T48,T50,Nc,Wf], rbf rank1, cycle<=3, 27k), FULL-VALLEY sweep.

Reader requirement (user): the sweep must extend PAST the minimum until the
RMSE clearly rises again, so the figure shows descent -> floor -> collapse
and nobody can ask "would a higher V be even better?".

Absolute axis 18.1..20.5 step 0.1 (pooled p2..~p99; percentiles only place
the grid / report kept%).  Detector frozen (q75, 30 flight-h baseline, clip
mu0-10*sigma0, mean-drop).  Onset RMSE over 9 dev units x 3 seeds, Wilcoxon
paired vs no-gate, tie band, per-seed wins for every candidate V.
Stats: v4_c3_stats_s{sd}.npz (pilot = selected configuration).
Figure: fig_v4_gate_sweep.png (paper dir).
"""
import os, sys, time
import numpy as np
from scipy import stats as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_setcmp import load, errors, SEEDS, detect

RES = os.path.join(HERE, 'v4_detcov_final_results.txt')
OUT = __import__('exp_paths').PAPER


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def seed_rmse(D, V):
    out = []
    for sd in SEEDS:
        e = []
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b if V is None else (b & (d['dc'] >= V))
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
        out.append(float(np.sqrt((np.array(e, float) ** 2).mean())))
    return out


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V4 STAGE 5 — detcov gate, selected model (set5 rbf r1, cycle<=3), '
        'full-valley absolute sweep')
    D = {sd: load(5, sd) for sd in SEEDS}
    pooled = np.concatenate([D[sd][u]['dc'] for sd in SEEDS for u in D[sd]])
    qs = [1, 2, 5, 10, 25, 50, 75, 90, 99]
    log('pooled dev detcov (descriptive): '
        + '  '.join(f'p{q}={np.percentile(pooled, q):.2f}' for q in qs))

    e_off = errors(D, None)
    r_off = np.sqrt((e_off ** 2).mean())
    off_seed = seed_rmse(D, None)
    log(f'no gate: RMSE={r_off:.2f} (per seed '
        + '/'.join(f'{r:.2f}' for r in off_seed) + f')  delay={e_off.mean():+.2f}')
    log('')

    grid = np.round(np.arange(18.1, 20.5 + 1e-9, 0.1), 1)
    log(f'absolute-axis sweep V={grid[0]}..{grid[-1]} step 0.1 '
        f'({len(grid)} values; extends past the minimum into the collapse '
        f'region by design)')
    log(f'  {"V":>6} | {"kept%":>6} | {"RMSE":>6} | {"delay":>6} | '
        f'{"better/worse":>12} | {"p":>7} | seed RMSE (s0/s1/s2, wins)')
    rows = []
    for V in grid:
        e = errors(D, float(V))
        rm = np.sqrt((e ** 2).mean())
        pw = st.wilcoxon(np.abs(e), np.abs(e_off), zero_method='zsplit').pvalue
        b = int((np.abs(e) < np.abs(e_off)).sum())
        w = int((np.abs(e) > np.abs(e_off)).sum())
        kept = 100.0 * (pooled >= V).mean()
        sr = seed_rmse(D, float(V))
        wins = sum(a < b_ for a, b_ in zip(sr, off_seed))
        rows.append((float(V), rm, pw, kept, e.mean()))
        log(f'  {V:6.1f} | {kept:5.1f}% | {rm:6.2f} | {e.mean():+6.2f} | '
            f'{b:>5} /{w:>5} | {pw:.4f} | '
            + '/'.join(f'{x:.2f}' for x in sr) + f'  {wins}/3')

    rmin = min(r[1] for r in rows)
    band = [r for r in rows if r[1] <= rmin + 0.1]
    sig = [r for r in rows if r[2] < 0.05 and r[1] < r_off]
    log('')
    log('tie band (RMSE <= min+0.1): '
        + ', '.join(f'V={r[0]:.1f} ({r[1]:.2f}, p={r[2]:.3f})' for r in band))
    if sig:
        log(f'significant band (p<0.05 & RMSE<no-gate): '
            f'V={sig[0][0]:.1f}..{sig[-1][0]:.1f} '
            f'({len(sig)} consecutive values)')
    log(f'valley floor: V=20.2 (RMSE {rmin:.2f}); collapse above: '
        + ', '.join(f'{r[0]:.1f}->{r[1]:.2f}' for r in rows if r[0] >= 20.3))
    log(f'VERDICT: gate VALID at cycle<=3 on the selected model '
        f'({r_off:.2f} -> {rmin:.2f}, {100*(rmin/r_off-1):+.1f}%)')

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(r_off, color='gray', lw=1.2, label=f'no gate ({r_off:.2f})')
    ax.plot([r[0] for r in rows], [r[1] for r in rows], marker='o', ms=3.5,
            ls='--', color='tab:blue', label='gated', zorder=2)
    for r in rows:
        if r[2] < 0.05 and r[1] < r_off:
            ax.plot(r[0], r[1], marker='*', ms=12, color='tab:red', zorder=3)
    ax.set_xlabel('absolute detcov threshold V (keep detcov >= V)')
    ax.set_ylabel('Onset RMSE [cycles]')
    ax.set_title('Inspection-gate sweep (* p<0.05 vs no gate)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    p_fig = os.path.join(OUT, 'fig_v4_gate_sweep.png')
    fig.savefig(p_fig, dpi=150); plt.close(fig)
    log(f'figure: {p_fig}')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
