"""
exp_v4_setcmp.py — v4 STAGE 4: sensor-set decision 3 vs 5 vs 7 (cycle<=3).

Winners per set (Stage 2+2b, complete-config rule): rbf rank1 everywhere.
Stats: set3 v4_stats_set3_s*, set5 v4_c3_stats_s* (pilot, identical config),
set7 v4_stats_set7_s*.  Flight hours shared (rul_input_s*).

IDENTICAL detection rule for every set (no per-set tuning):
  cycle summary q75 of corrected LL, 30 flight-hour baseline,
  clip mu0-10*sigma0, mean-drop.
Gate: each set gets its own ABSOLUTE detcov sweep (0.1 grid spanning pooled
p2..p90 so the valley floor AND the right-side rise are visible; percentiles
only place the grid / report kept%).  Reported per set: gate-off RMSE,
sweep-minimum RMSE with its V and tie band (min+0.1).
Selection: min gated RMSE; pairwise Wilcoxon on |error| at each set's best V.
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import meandrop

SEEDS = [0, 1, 2]
FILES = {3: 'v4_stats_set3_s{sd}.npz', 5: 'v4_c3_stats_s{sd}.npz',
         7: 'v4_stats_set7_s{sd}.npz'}
RES = os.path.join(HERE, 'v4_setcmp_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load(setn, sd):
    Z = np.load(os.path.join(HERE, FILES[setn].format(sd=sd)))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files
                    if k.endswith('_cc')})
    out = {}
    for u in units:
        cc = Z[f'u{u}_cc']; ll = Z[f'u{u}_ll']; dc = Z[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        out[u] = dict(cc=cc, ll=ll, dc=dc, ucyc=ucyc,
                      dur=np.array([durs[pos[int(c)]] for c in ucyc], float),
                      onset=int(Z[f'u{u}_onset'][0]))
    return out


def detect(cur, dur, ucyc):
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def errors(D, V):
    e = []
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b if V is None else (b & (d['dc'] >= V))
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V4 STAGE 4 — set decision (identical rule: q75, 30h baseline, '
        'clip 10s; per-set absolute sweep)')
    R = {}
    for setn in [3, 5, 7]:
        D = {sd: load(setn, sd) for sd in SEEDS}
        pooled = np.concatenate([D[sd][u]['dc'] for sd in SEEDS for u in D[sd]])
        e_off = errors(D, None)
        r_off = np.sqrt((e_off ** 2).mean())
        lo = np.floor(np.percentile(pooled, 2) * 10) / 10
        hi = np.ceil(np.percentile(pooled, 90) * 10) / 10
        grid = np.round(np.arange(lo, hi + 1e-9, 0.1), 1)
        rows = []
        for V in grid:
            e = errors(D, float(V))
            rows.append((float(V), np.sqrt((e ** 2).mean()),
                         100.0 * (pooled >= V).mean(), e))
        rmin = min(r[1] for r in rows)
        band = [r for r in rows if r[1] <= rmin + 0.1]
        bestV, bestR, bestK, bestE = band[-1] if False else \
            [r for r in rows if r[1] == rmin][0]
        pw = st.wilcoxon(np.abs(bestE), np.abs(e_off),
                         zero_method='zsplit').pvalue
        R[setn] = dict(off=e_off, r_off=r_off, best=(bestV, bestR, bestK),
                       bestE=bestE, band=band, grid=(grid[0], grid[-1]))
        log(f'  set{setn}: gate-off RMSE={r_off:5.2f} | sweep '
            f'V={grid[0]}..{grid[-1]} -> min RMSE={bestR:.2f} at V={bestV} '
            f'(kept {bestK:.0f}%, p={pw:.4f} vs off)')
        log(f'          tie band (<=min+0.1): '
            + ', '.join(f'{r[0]:.1f}({r[1]:.2f})' for r in band))

    log('')
    gated = {n: R[n]['best'][1] for n in R}
    win = min(gated, key=gated.get)
    others = [n for n in R if n != win]
    for b in others:
        ea, eb = np.abs(R[win]['bestE']), np.abs(R[b]['bestE'])
        p = st.wilcoxon(ea, eb, zero_method='zsplit').pvalue
        log(f'  set{win} vs set{b} (each at own best V, paired 27): better '
            f'{int((ea < eb).sum())} worse {int((ea > eb).sum())}  '
            f'Wilcoxon p={p:.3f}')
    log('')
    log(f'SELECTED SET: {win}  (gated RMSE {gated[win]:.2f}; others '
        f'{sorted((n, round(v, 2)) for n, v in gated.items() if n != win)}; '
        f'gate-off ranking {sorted((n, round(R[n]["r_off"], 2)) for n in R)})')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
