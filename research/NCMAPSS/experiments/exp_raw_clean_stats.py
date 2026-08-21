"""
exp_raw_clean_stats.py — PHASE 1 analysis: paired test of cleaning against the
MATCHED random-removal control.

Plain zRMSE is dominated by (i) the evaluation set's own outliers, which no
training-side cleaning can remove, and (ii) the draw-to-draw variance of the
8192-point training sample.  The robust metrics (trimmed zRMSE over the central
90 %, and median |z|) isolate the bulk of the distribution, which is what
cleaner training data should improve.  Each cleaning cut is paired with the
random removal of the SAME size within the SAME seed, so the draw variance and
the seed effect both cancel.
"""
import numpy as np
from scipy import stats as st

# per-seed values transcribed from raw_clean_results.txt (seeds 0,1,2)
D = {
    'baseline':    dict(trim=[.4118, .4053, .4072], med=[.3319, .3167, .3217]),
    'clean>5':     dict(trim=[.3743, .3631, .3719], med=[.3081, .2924, .3108]),
    'random3.7%':  dict(trim=[.4159, .3589, .4238], med=[.3298, .2971, .3429]),
    'clean>3':     dict(trim=[.3884, .3468, .3972], med=[.3237, .2913, .3146]),
    'random6.8%':  dict(trim=[.4244, .3744, .4099], med=[.3311, .3114, .3270]),
    'clean>2':     dict(trim=[.3977, .3514, .3779], med=[.3266, .2966, .3122]),
    'random10.8%': dict(trim=[.4017, .3629, .4578], med=[.3324, .3025, .3571]),
}
PAIRS = [('clean>5', 'random3.7%', 3.7),
         ('clean>3', 'random6.8%', 6.8),
         ('clean>2', 'random10.8%', 10.8)]

print('PHASE 1 — cleaning vs MATCHED random removal (paired within seed)\n')
for metric in ['trim', 'med']:
    nm = 'trimmed zRMSE (central 90%)' if metric == 'trim' else 'median |z|'
    print(f'=== {nm} ===')
    base = np.array(D['baseline'][metric])
    print(f'  baseline: {base.mean():.4f} ± {base.std():.4f}')
    wins = []
    for c, r, rate in PAIRS:
        cv, rv = np.array(D[c][metric]), np.array(D[r][metric])
        d = rv - cv                       # >0 means cleaning is better
        wins += list(d > 0)
        print(f'  {c:>8} ({rate:4.1f}%): {cv.mean():.4f}  vs random {rv.mean():.4f}  '
              f'| per-seed gain {np.round(d,4)}  | wins {int((d>0).sum())}/3  '
              f'| vs baseline {100*(cv.mean()-base.mean())/base.mean():+.1f}%')
    k, n = int(np.sum(wins)), len(wins)
    p = st.binomtest(k, n, 0.5).pvalue
    print(f'  --> cleaning beats matched random in {k}/{n} paired comparisons, '
          f'sign test p = {p:.4f}')
    allc = np.concatenate([D[c][metric] for c, _, _ in PAIRS])
    allr = np.concatenate([D[r][metric] for _, r, _ in PAIRS])
    w = st.wilcoxon(allc, allr).pvalue
    print(f'      Wilcoxon signed-rank p = {w:.4f}   '
          f'mean gain = {100*(allr.mean()-allc.mean())/allr.mean():+.1f}%\n')

print('random-removal controls vs baseline (should be ~no effect):')
for metric in ['trim', 'med']:
    base = np.array(D['baseline'][metric])
    allr = np.concatenate([D[r][metric] for _, r, _ in PAIRS])
    print(f'  {metric:>4}: baseline {base.mean():.4f}  random {allr.mean():.4f}  '
          f'({100*(allr.mean()-base.mean())/base.mean():+.1f}%)')
