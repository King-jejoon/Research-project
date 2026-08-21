"""
exp_v3_a2cmp.py — STAGE A2 (part 2): sensor-set decision 3 vs 5 vs 7.

Every set carries its own Stage-A winner (rbf rank1) trained on the identical
stratified split; per-point stats come from
  set3: v3_stats_set3_s{seed}.npz     set7: v3_stats_set7_s{seed}.npz
  set5: v2_stats_s{seed}.npz          (identical configuration, reused)
Flight-hours per cycle are shared across sets (same flights): rul_input_s*.

IDENTICAL detection rule for every set (no per-set tuning):
  cycle summary = q75 of the corrected LL
  detector      = 30 flight-hour baseline -> clip mu0-10*sigma0 -> mean-drop
  two gate variants, both applied to every set the same way:
    off      : all 200 points
    V@p25    : detcov >= 25th pct of that set's pooled dev detcov (absolute)
Score: onset RMSE, 9 dev units x 3 seeds.  Selection: min RMSE under the gated
rule (the gate is part of the validated chain in this data regime); the
ungated column is reported as a robustness check.
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import meandrop

SEEDS = [0, 1, 2]
FILES = {3: 'v3_stats_set3_s{sd}.npz', 5: 'v2_stats_s{sd}.npz',
         7: 'v3_stats_set7_s{sd}.npz'}
RES = os.path.join(HERE, 'v3_a2cmp_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load(setn, sd):
    Z = np.load(os.path.join(HERE, FILES[setn].format(sd=sd)))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files if k.endswith('_cc')})
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
                m = b if V is None else (b & (d['dc'] >= V[sd]))
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            e.append(detect(cur, d['dur'], d['ucyc']) - d['onset'])
    return np.array(e, float)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('A2 — sensor-set decision (identical rule: q75, 30h baseline, clip 10s)')
    R = {}
    for setn in [3, 5, 7]:
        D = {sd: load(setn, sd) for sd in SEEDS}
        V = {sd: float(np.percentile(
            np.concatenate([D[sd][u]['dc'] for u in D[sd]]), 25)) for sd in SEEDS}
        e_off = errors(D, None)
        e_gate = errors(D, V)
        R[setn] = (e_off, e_gate, V)
        log(f'  set{setn}: gate-off RMSE={np.sqrt((e_off**2).mean()):5.2f} '
            f'(delay {e_off.mean():+5.2f}) | gate@p25 '
            f'RMSE={np.sqrt((e_gate**2).mean()):5.2f} '
            f'(delay {e_gate.mean():+5.2f})  V*={[round(V[s],2) for s in SEEDS]}')

    log('')
    gated = {n: np.sqrt((R[n][1] ** 2).mean()) for n in R}
    win = min(gated, key=gated.get)
    for a, b in [(win, [n for n in R if n != win][0]),
                 (win, [n for n in R if n != win][1])]:
        ea, eb = np.abs(R[a][1]), np.abs(R[b][1])
        p = st.wilcoxon(ea, eb, zero_method='zsplit').pvalue
        log(f'  set{a} vs set{b} (gated, paired 27): better '
            f'{int((ea<eb).sum())} worse {int((ea>eb).sum())}  Wilcoxon p={p:.3f}')
    log('')
    log(f'SELECTED SET: {win}  (gated RMSE {gated[win]:.2f}; '
        f'others {sorted((n, round(v,2)) for n, v in gated.items() if n != win)})')
    log(f'\nwall={(time.time()-t0)/60:.2f} min')


if __name__ == '__main__':
    main()
