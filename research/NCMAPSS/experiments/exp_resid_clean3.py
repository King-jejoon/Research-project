"""
exp_resid_clean3.py — PHASE 2c: which per-point quantity should the detector
summarise, and does cleaning add anything on top?

  LL = detcov - 0.5*m2 - c
       \_______/   \____/
        y-blind     the only health-dependent part

The frozen chain tracks a quantile of LL, so every cycle statistic carries the
detcov term, which reacts to where the flight happened to sample the operating
envelope and not to engine health.  Using m2 alone should remove that nuisance.
Degradation inflates m2, and the change-point rule looks for a mean DROP, so
the statistic is -m2.

Swept: statistic {LL, -m2} x quantile x point filter {none, detcov, within-cycle
robust}.  Paired Wilcoxon against the current chain (detcov gate + q75 of LL) on
the 27 dev cases, because the configurations differ on only a handful of units.
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import load, detect, SEEDS

QS = [60, 70, 75, 80, 85, 90, 95]
FILTERS = [('none',), ('detcov',), ('wz', 3.0)]
RES = os.path.join(HERE, 'resid_clean3_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def curve(d, stat, filt, qq, Vstar):
    cc, ll, dc, m2 = d['cc'], d['ll'], d['dc'], d['m2']
    val = ll if stat == 'LL' else -m2
    out = np.empty(len(d['ucyc']))
    for i, c in enumerate(d['ucyc']):
        b = cc == c
        v, vd, vv = val[b], dc[b], val[b]
        if filt[0] == 'none':
            m = np.ones(len(v), bool)
        elif filt[0] == 'detcov':
            m = vd >= Vstar
        else:
            med = np.median(vv)
            mad = np.median(np.abs(vv - med)) * 1.4826 + 1e-9
            m = np.abs(vv - med) / mad <= filt[1]
        if not m.any():
            m = np.ones(len(v), bool)
        out[i] = np.percentile(v[m], qq)
    return out


def main():
    open(RES, 'w').close()
    t0 = time.time()
    D = {sd: load(sd) for sd in SEEDS}
    Vs = {sd: float(np.percentile(np.concatenate([D[sd][u]['dc'] for u in D[sd]]), 15))
          for sd in SEEDS}
    log('PHASE 2c — detection statistic: LL (mixes detcov) vs -m2 (health only)')
    log('  detector frozen: 30 flight-hour baseline, clip 10 sigma, mean-drop')

    err = {}
    for stat in ['LL', '-m2']:
        log('')
        log(f'  statistic = {stat}')
        log(f'{"filter":>10} | ' + ' | '.join(f'q{q:<4}' for q in QS))
        log('  ' + '-' * 66)
        for f in FILTERS:
            fn = '+'.join(map(str, f))
            row = []
            for q in QS:
                e = []
                for sd in SEEDS:
                    for u, d in D[sd].items():
                        e.append(detect(curve(d, stat, f, q, Vs[sd]),
                                        d['dur'], d['ucyc']) - d['onset'])
                err[(stat, fn, q)] = np.array(e, float)
                row.append(np.sqrt((err[(stat, fn, q)] ** 2).mean()))
            log(f'{fn:>10} | ' + ' | '.join(f'{v:5.2f}' for v in row))

    ref = err[('LL', 'detcov', 75)]
    log('')
    log('paired comparison against the current chain (LL + detcov gate + q75, '
        f'RMSE={np.sqrt((ref**2).mean()):.2f}) over the 27 dev cases:')
    log(f'{"config":>24} | {"RMSE":>5} | {"s0/s1/s2":>16} | {"better":>6} | '
        f'{"worse":>5} | {"Wilcoxon p":>10}')
    log('-' * 86)
    for k in sorted(err, key=lambda k: np.sqrt((err[k] ** 2).mean()))[:8]:
        e = err[k]
        per = [np.sqrt((e[i * 9:(i + 1) * 9] ** 2).mean()) for i in range(3)]
        b = int((np.abs(e) < np.abs(ref)).sum()); w = int((np.abs(e) > np.abs(ref)).sum())
        try:
            p = st.wilcoxon(np.abs(e), np.abs(ref), zero_method='zsplit').pvalue
        except ValueError:
            p = np.nan
        log(f'{k[0]+" "+k[1]+" q"+str(k[2]):>24} | {np.sqrt((e**2).mean()):5.2f} | '
            f'{per[0]:4.2f}/{per[1]:4.2f}/{per[2]:4.2f} | {b:>6} | {w:>5} | {p:10.3f}')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
