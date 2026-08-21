"""
exp_v3_b_detcov.py — STAGE B: the covariance (detcov) experiment on the
winning model — find the best improvement and its significance.

Gate rule: keep points with detcov >= V, V anchored at pooled-dev percentiles
{5,10,15,20,25,30,40} and reported as ABSOLUTE values (transferable).  Cycle
summary q75, detector frozen (30 flight-hour baseline, clip 10 sigma,
mean-drop).  Score: state-identification (onset) RMSE over 9 dev units x 3
seeds; every V is paired against the no-gate baseline (Wilcoxon on |error|).
Output: improvement curve figure + selected V* (most conservative V whose
improvement is significant at p<0.05; if none is significant, the minimum-RMSE
V is reported but flagged).
SET env selects the winner set (default 5).
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v3_a2cmp import load, detect, SEEDS

SET = int(os.environ.get('SET', 5))
PCTS = [5, 10, 15, 20, 25, 30, 40]
RES = os.path.join(HERE, 'v3_b_detcov_results.txt')
OUT = __import__('exp_paths').PAPER


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def errors(D, V):
    e = []
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b if V is None else (b & (d['dc'] >= V[sd]))
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'STAGE B — detcov gate sweep on set{SET} (q75, 30h baseline, clip 10s)')
    D = {sd: load(SET, sd) for sd in SEEDS}
    pooled = {sd: np.concatenate([D[sd][u]['dc'] for u in D[sd]]) for sd in SEEDS}
    e_off = errors(D, None)
    r_off = np.sqrt((e_off ** 2).mean())
    log(f'  no gate: RMSE={r_off:.2f}  delay={e_off.mean():+.2f}')
    log('')
    log(f'  {"pct":>4} | {"abs V (s0/s1/s2)":>24} | {"kept%":>6} | {"RMSE":>6} | '
        f'{"delay":>6} | {"better/worse":>12} | p')
    rows = []
    sig = []
    for p_ in PCTS:
        V = {sd: float(np.percentile(pooled[sd], p_)) for sd in SEEDS}
        kept = np.mean([100 - p_])
        e = errors(D, V)
        rm = np.sqrt((e ** 2).mean())
        pw = st.wilcoxon(np.abs(e), np.abs(e_off), zero_method='zsplit').pvalue
        b = int((np.abs(e) < np.abs(e_off)).sum())
        w = int((np.abs(e) > np.abs(e_off)).sum())
        rows.append((p_, rm, pw))
        if pw < 0.05 and rm < r_off:
            sig.append(p_)
        log(f'  {p_:>4} | {V[0]:7.2f}/{V[1]:7.2f}/{V[2]:7.2f} | {100-p_:>5}% | '
            f'{rm:6.2f} | {e.mean():+6.2f} | {b:>5} /{w:>5} | {pw:.4f}')
    log('')
    if sig:
        vstar_p = min(sig)                      # most conservative significant V
        note = 'most conservative significant V'
    else:
        vstar_p = min(rows, key=lambda r: r[1])[0]
        note = 'NO significant V; min-RMSE reported with flag'
    Vst = {sd: float(np.percentile(pooled[sd], vstar_p)) for sd in SEEDS}
    rm = [r for r in rows if r[0] == vstar_p][0][1]
    log(f'V* = pct{vstar_p} (abs {[round(Vst[s],2) for s in SEEDS]})  '
        f'RMSE {rm:.2f} vs no-gate {r_off:.2f}  [{note}]')

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.axhline(r_off, color='gray', lw=1.2, label=f'no gate ({r_off:.2f})')
    xs = [np.mean([np.percentile(pooled[sd], p_) for sd in SEEDS]) for p_ in PCTS]
    ax.plot(xs, [r[1] for r in rows], marker='o', ls='--', color='tab:blue')
    for x, (p_, rm_, pw) in zip(xs, rows):
        if pw < 0.05:
            ax.plot(x, rm_, marker='*', ms=14, color='tab:red')
    ax.set_xlabel('absolute detcov threshold  V  (keep detcov >= V)')
    ax.set_ylabel('Onset RMSE [cycles]')
    ax.set_title(f'Inspection-gate sweep (set{SET}; * = p<0.05 vs no gate)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(frameon=False)
    fig.tight_layout()
    p_fig = os.path.join(OUT, 'fig_v3_gate_sweep.png')
    fig.savefig(p_fig, dpi=150); plt.close(fig)
    log(f'figure: {p_fig}')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
