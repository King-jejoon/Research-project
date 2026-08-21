"""
exp_resid_clean.py — PHASE 2: cleaning the RESIDUAL (per-point) data, scored by
how well the 30-flight-hour-baseline detector finds the hs inflection.

Point-level data is reused from sensor5_stats_s{seed}.npz, which was produced
with EXACTLY the frozen configuration (5 sensors T30/T48/T50/Nc/Wf, RBF,
coregionalization rank 1, Vecchia m=18, Adam lr 0.1 x 120, stage-1 detcov 5 %
cleaning, 200 points per cycle, per-unit rng u*7+seed).  Flight durations come
from rul_input_s{seed}.npz.  No GP is refitted, so the whole sweep is seconds.

Per point we have LL (corrected whitening) and detcov; the Mahalanobis term is
recovered as  m2 = 2*(detcov - LL - c),  c = (q/2)log(2*pi), q = 5.

POINT FILTERS (the "cleaning")
  none         keep all 200 points of the cycle
  detcov>=V    the current inspection gate, absolute V (y-blind)
  m2<=T        global cut on the Mahalanobis term.  WARNING: after onset the
               degradation itself inflates m2, so a global cut deletes signal;
               included to demonstrate exactly that.
  wz<=k        WITHIN-CYCLE robust cut: |LL - median_cycle| / MAD_cycle <= k.
               Degradation shifts the whole cycle, so this removes measurement
               outliers while preserving the health-driven level change — the
               principled way to clean residual data for detection.
  wz&detcov    both

CYCLE SUMMARIES: mean, median, q75 (current), q90, trim25
DETECTOR (frozen): baseline window = first 30 flight-hours, clip at
mu0 - 10*sigma0, full-range mean-drop change point.
SCORE: onset RMSE over 9 dev units x 3 seeds (27 cases), mean signed delay,
and the fraction of cases within +-3 cycles.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SEEDS = [0, 1, 2]
Q = 5
CONST = 0.5 * Q * np.log(2 * np.pi)
RES = os.path.join(HERE, 'resid_clean_results.txt')
NPZ = os.path.join(HERE, 'resid_clean.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load(sd):
    """per-unit: cc, ll, dc, m2, onset, ucyc, dur(flight hours per cycle)."""
    P = np.load(os.path.join(HERE, f'sensor5_stats_s{sd}.npz'))
    R = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in P.files if k.endswith('_cc')})
    out = {}
    for u in units:
        cc = P[f'u{u}_cc']; ll = P[f'u{u}_ll']; dc = P[f'u{u}_dc']
        ucyc_r = R[f'dev_{u}_ucyc']; dur = R[f'dev_{u}_hours']
        ucyc = np.unique(cc)
        # align the flight-duration table to the cycles present in the point data
        pos = {int(c): i for i, c in enumerate(ucyc_r)}
        keep = np.array([int(c) in pos for c in ucyc])
        ucyc = ucyc[keep]
        d = np.array([dur[pos[int(c)]] for c in ucyc], float)
        out[u] = dict(cc=cc, ll=ll, dc=dc, m2=2.0 * (dc - ll - CONST),
                      ucyc=ucyc, dur=d, onset=int(P[f'u{u}_onset'][0]))
    return out


def meandrop(x, kmin=3):
    n = len(x); s = np.std(x, ddof=1) + 1e-12; cs = np.cumsum(x)
    best = (None, -np.inf)
    for k in range(kmin, n - 4):
        m1 = cs[k - 1] / k; m2_ = (cs[-1] - cs[k - 1]) / (n - k)
        t = np.sqrt(k * (n - k) / n) * (m1 - m2_) / s
        if t > best[1]:
            best = (k, t)
    return best[0]


def detect(curve, dur, ucyc):
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0, sd0 = curve[:w].mean(), curve[:w].std(ddof=1) + 1e-8
    x = np.maximum(curve, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def summarize(v, how):
    if how == 'mean':   return v.mean()
    if how == 'median': return np.median(v)
    if how == 'q75':    return np.percentile(v, 75)
    if how == 'q90':    return np.percentile(v, 90)
    if how == 'trim25':
        s = np.sort(v); n = len(s)
        return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean()
    raise ValueError(how)


def curve_of(d, filt, how, Vstar):
    """per-cycle summary of the gated LL curve."""
    cc, ll, dc, m2 = d['cc'], d['ll'], d['dc'], d['m2']
    vals = np.empty(len(d['ucyc']))
    for i, c in enumerate(d['ucyc']):
        base = cc == c
        v_ll, v_dc, v_m2 = ll[base], dc[base], m2[base]
        if filt[0] == 'none':
            m = np.ones(len(v_ll), bool)
        elif filt[0] == 'detcov':
            m = v_dc >= Vstar
        elif filt[0] == 'm2':
            m = v_m2 <= filt[1]
        elif filt[0] == 'wz':
            med = np.median(v_ll)
            mad = np.median(np.abs(v_ll - med)) * 1.4826 + 1e-9
            m = np.abs(v_ll - med) / mad <= filt[1]
        elif filt[0] == 'wz+detcov':
            med = np.median(v_ll)
            mad = np.median(np.abs(v_ll - med)) * 1.4826 + 1e-9
            m = (np.abs(v_ll - med) / mad <= filt[1]) & (v_dc >= Vstar)
        if not m.any():
            m = np.ones(len(v_ll), bool)
        vals[i] = summarize(v_ll[m], how)
    return vals


def main():
    open(RES, 'w').close()
    t0 = time.time()
    D = {sd: load(sd) for sd in SEEDS}
    Vstar = {sd: float(np.percentile(np.concatenate([D[sd][u]['dc']
                                                     for u in D[sd]]), 15))
             for sd in SEEDS}
    log('PHASE 2 — residual-data cleaning scored by onset detection')
    log('  detector frozen: 30 flight-hour baseline, clip mu0-10*sigma0, mean-drop')
    log(f'  dev units={sorted(D[0])}  seeds={SEEDS}')
    log(f'  V* (15th pct of pooled dev detcov) per seed: '
        f'{[round(Vstar[s],3) for s in SEEDS]}')
    kept = {}
    FILTERS = [('none',), ('detcov',),
               ('m2', 15.09), ('m2', 30.0),
               ('wz', 4.0), ('wz', 3.0), ('wz', 2.5),
               ('wz+detcov', 3.0)]
    HOWS = ['mean', 'median', 'q75', 'q90', 'trim25']

    # how many points each filter keeps (sanity)
    for f in FILTERS:
        tot = k = 0
        for sd in SEEDS:
            for u, d in D[sd].items():
                for c in d['ucyc']:
                    b = d['cc'] == c
                    if f[0] == 'none':      m = np.ones(b.sum(), bool)
                    elif f[0] == 'detcov':  m = d['dc'][b] >= Vstar[sd]
                    elif f[0] == 'm2':      m = d['m2'][b] <= f[1]
                    else:
                        v = d['ll'][b]; med = np.median(v)
                        mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
                        m = np.abs(v - med) / mad <= f[1]
                        if f[0] == 'wz+detcov':
                            m &= d['dc'][b] >= Vstar[sd]
                    tot += b.sum(); k += m.sum()
        kept['+'.join(map(str, f))] = 100.0 * k / tot

    log('')
    log(f'{"filter":>14} | kept% | ' + ' | '.join(f'{h:>7}' for h in HOWS))
    log('-' * 72)
    best = (None, np.inf)
    table = {}
    for f in FILTERS:
        fname = '+'.join(map(str, f))
        row = []
        for how in HOWS:
            errs = []
            for sd in SEEDS:
                for u, d in D[sd].items():
                    cur = curve_of(d, f, how, Vstar[sd])
                    errs.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
            errs = np.array(errs, float)
            rmse = float(np.sqrt((errs ** 2).mean()))
            table[(fname, how)] = (rmse, float(errs.mean()),
                                   float((np.abs(errs) <= 3).mean()))
            row.append(rmse)
            if rmse < best[1]:
                best = ((fname, how), rmse)
        log(f'{fname:>14} | {kept[fname]:5.1f} | ' +
            ' | '.join(f'{v:7.2f}' for v in row))

    log('')
    (bf, bh), br = best
    d_, dl, hit = table[(bf, bh)]
    log(f'BEST: filter={bf}  summary={bh}  onset RMSE={br:.2f} cycles  '
        f'(mean delay {dl:+.2f}, |d|<=3 in {100*hit:.0f}%)')
    cur = table[('detcov', 'q75')]
    log(f'CURRENT frozen chain (detcov gate + q75): RMSE={cur[0]:.2f}  '
        f'(mean delay {cur[1]:+.2f}, |d|<=3 in {100*cur[2]:.0f}%)')
    nf = table[('none', 'q75')]
    log(f'NO point filter (q75)                   : RMSE={nf[0]:.2f}  '
        f'(mean delay {nf[1]:+.2f}, |d|<=3 in {100*nf[2]:.0f}%)')
    np.savez(NPZ, **{f'{a}|{b}': np.array(v) for (a, b), v in table.items()})
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
