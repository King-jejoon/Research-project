"""
exp_clean_compare.py — PHASE 1c: does the validated raw-data cleaning carry
through to onset detection?

Two final models differing ONLY in the cleaning step (same sensors, kernel,
rank, Vecchia m, optimiser, candidate draw rng 2000+seed, final draw rng
3000+seed, per-unit sampling rng unit*7+seed, 200 points per cycle):

  detcov   sensor5_stats_s*.npz   stage-1 GP -> drop the lowest 5 % detcov
  rawclean newclean_stats_s*.npz  model-free robust neighbour score > 5 removed

Scored with the frozen detector: per-cycle q75 of LL -> baseline window = first
30 flight-hours -> clip at mu0 - 10*sigma0 -> full-range mean-drop, against the
hs-transition cycle, over 9 dev units x 3 seeds.  Paired per unit and seed, so
the comparison is like for like.  Also reports the summary-statistic sweep so
the two cleanings are compared over a range of detector settings, not a single
lucky one.
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import detect, CONST, SEEDS

QS = [70, 75, 80, 85, 90]
RES = os.path.join(HERE, 'clean_compare_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load(tag, sd):
    fn = {'detcov': f'sensor5_stats_s{sd}.npz',
          'rawclean': f'newclean_stats_s{sd}.npz'}[tag]
    P = np.load(os.path.join(HERE, fn))
    R = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in P.files if k.endswith('_cc')})
    out = {}
    for u in units:
        cc = P[f'u{u}_cc']; ll = P[f'u{u}_ll']; dc = P[f'u{u}_dc']
        ucyc_r = R[f'dev_{u}_ucyc']; dur = R[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ucyc_r)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        out[u] = dict(cc=cc, ll=ll, dc=dc, ucyc=ucyc,
                      dur=np.array([dur[pos[int(c)]] for c in ucyc], float),
                      onset=int(P[f'u{u}_onset'][0]))
    return out


def run(D, Vs, q, gate):
    e = []
    for sd in SEEDS:
        for u, d in D[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                v = d['ll'][b]
                m = (d['dc'][b] >= Vs[sd]) if gate else np.ones(len(v), bool)
                if not m.any():
                    m = np.ones(len(v), bool)
                cur[i] = np.percentile(v[m], q)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    D = {t: {sd: load(t, sd) for sd in SEEDS} for t in ['detcov', 'rawclean']}
    Vs = {t: {sd: float(np.percentile(np.concatenate(
        [D[t][sd][u]['dc'] for u in D[t][sd]]), 15)) for sd in SEEDS}
        for t in D}
    log('PHASE 1c — onset detection: detcov cleaning vs raw-data cleaning')
    log('  detector frozen: q of LL -> 30 flight-hour baseline -> clip 10 sigma '
        '-> mean-drop;  9 dev units x 3 seeds')
    for t in D:
        log(f'  V* (15th pct pooled detcov, {t}): '
            f'{[round(Vs[t][s],3) for s in SEEDS]}')

    log('')
    log(f'{"cleaning":>10} | {"gate":>6} | ' + ' | '.join(f'q{q:<4}' for q in QS))
    log('-' * 62)
    E = {}
    for t in ['detcov', 'rawclean']:
        for gate in [True, False]:
            row = []
            for q in QS:
                E[(t, gate, q)] = run(D[t], Vs[t], q, gate)
                row.append(np.sqrt((E[(t, gate, q)] ** 2).mean()))
            log(f'{t:>10} | {str(gate):>6} | ' + ' | '.join(f'{v:5.2f}' for v in row))

    log('')
    log('paired comparison at the frozen setting (q75 + detcov gate):')
    a, b = E[('rawclean', True, 75)], E[('detcov', True, 75)]
    pa = [np.sqrt((a[i * 9:(i + 1) * 9] ** 2).mean()) for i in range(3)]
    pb = [np.sqrt((b[i * 9:(i + 1) * 9] ** 2).mean()) for i in range(3)]
    log(f'  detcov  cleaning: RMSE={np.sqrt((b**2).mean()):.2f}  per-seed '
        f'{np.round(pb,2)}  mean delay {b.mean():+.2f}')
    log(f'  rawdata cleaning: RMSE={np.sqrt((a**2).mean()):.2f}  per-seed '
        f'{np.round(pa,2)}  mean delay {a.mean():+.2f}')
    better = int((np.abs(a) < np.abs(b)).sum()); worse = int((np.abs(a) > np.abs(b)).sum())
    try:
        p = st.wilcoxon(np.abs(a), np.abs(b), zero_method='zsplit').pvalue
    except ValueError:
        p = float('nan')
    log(f'  rawclean better in {better} cases, worse in {worse}, '
        f'tied in {27-better-worse}  | Wilcoxon p={p:.3f}')

    log('')
    log('per-unit signed error (rawclean / detcov), seed 0 | 1 | 2:')
    units = sorted(D['detcov'][0])
    for j, u in enumerate(units):
        log(f'  unit {u:>2} | ' + ' | '.join(
            f'{a[s*9+j]:+4.0f} /{b[s*9+j]:+4.0f}' for s in range(3)))
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
