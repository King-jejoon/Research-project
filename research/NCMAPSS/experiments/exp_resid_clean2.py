"""
exp_resid_clean2.py — PHASE 2b: refine the summary/filter choice and check that
it is stable across seeds (not a lucky pick on 27 cases).

Phase 2 showed the ORDER STATISTIC dominates: with a fragile summary (mean)
point cleaning buys a lot, with a robust one (q75/q90) it buys nothing, and the
overall best was no filter at all with q90.  Here the LL quantile is swept
finely, crossed with the main filters, and the top candidates are reported
per seed and per unit so the winner can be judged against seed noise.
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import load, detect, curve_of, SEEDS

QS = [60, 65, 70, 75, 80, 85, 90, 95]
FILTERS = [('none',), ('detcov',), ('wz', 3.0), ('wz+detcov', 3.0)]
RES = os.path.join(HERE, 'resid_clean2_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def curve_q(d, filt, qq, Vstar):
    cc, ll, dc, m2 = d['cc'], d['ll'], d['dc'], d['m2']
    out = np.empty(len(d['ucyc']))
    for i, c in enumerate(d['ucyc']):
        b = cc == c
        v, vd = ll[b], dc[b]
        if filt[0] == 'none':
            m = np.ones(len(v), bool)
        elif filt[0] == 'detcov':
            m = vd >= Vstar
        else:
            med = np.median(v)
            mad = np.median(np.abs(v - med)) * 1.4826 + 1e-9
            m = np.abs(v - med) / mad <= filt[1]
            if filt[0] == 'wz+detcov':
                m &= vd >= Vstar
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
    log('PHASE 2b — LL-quantile sweep x point filter  (onset RMSE, cycles)')
    log('  detector frozen: 30 flight-hour baseline, clip 10 sigma, mean-drop')
    log('')
    log(f'{"filter":>14} | ' + ' | '.join(f'q{q:<4}' for q in QS))
    log('-' * 78)
    err = {}
    for f in FILTERS:
        fn = '+'.join(map(str, f))
        row = []
        for q in QS:
            e = []
            for sd in SEEDS:
                for u, d in D[sd].items():
                    e.append(detect(curve_q(d, f, q, Vs[sd]), d['dur'], d['ucyc'])
                             - d['onset'])
            err[(fn, q)] = np.array(e, float)
            row.append(np.sqrt((err[(fn, q)] ** 2).mean()))
        log(f'{fn:>14} | ' + ' | '.join(f'{v:5.2f}' for v in row))

    log('')
    log('per-seed RMSE of the leading candidates (9 units each):')
    log(f'{"config":>22} | {"s0":>5} | {"s1":>5} | {"s2":>5} | {"all":>5} | '
        f'{"mean d":>7} | {"|d|<=3":>6}')
    log('-' * 72)
    cands = sorted(err, key=lambda k: np.sqrt((err[k] ** 2).mean()))[:6]
    for k in list(dict.fromkeys(cands + [('detcov', 75), ('none', 75)])):
        e = err[k]
        per = [np.sqrt((e[i * 9:(i + 1) * 9] ** 2).mean()) for i in range(3)]
        log(f'{k[0]+" q"+str(k[1]):>22} | ' + ' | '.join(f'{v:5.2f}' for v in per) +
            f' | {np.sqrt((e**2).mean()):5.2f} | {e.mean():+7.2f} | '
            f'{100*(np.abs(e)<=3).mean():5.0f}%')

    log('')
    best = min(err, key=lambda k: np.sqrt((err[k] ** 2).mean()))
    log(f'per-unit signed error, best = {best[0]} q{best[1]} '
        f'vs current = detcov q75:')
    log(f'{"unit":>5} | ' + ' | '.join(f'{"s"+str(s):>10}' for s in SEEDS))
    eb, ec = err[best], err[('detcov', 75)]
    units = sorted(D[0])
    for j, u in enumerate(units):
        log(f'{u:>5} | ' + ' | '.join(
            f'{eb[s*9+j]:+4.0f} /{ec[s*9+j]:+4.0f}' for s in range(3)))
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
