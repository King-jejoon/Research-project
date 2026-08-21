"""
exp_v4_gate_redesign.py — candidate feasibility scan: gate mechanisms that
help BOTH flight classes on dev (one absolute V provably trades them off:
short/med optimum V=20.2 vs long optimum 18.8-19.2).

Candidates, all label-free at deployment and re-scored from cached stats:
  A. coverage-conditional V: baseline window cycle count n < NSPARSE
     -> V_sparse, else V_dense.  Grid over (V_dense, V_sparse).
  B. robust sigma0: baseline noise scale = 1.4826 * MAD instead of std
     (single flipped baseline cycle cannot inflate sigma0), x V sweep.
  C. combined A+B for the best A cell.
Scoring: dev 27 cases — overall / short+med / long (u6, u8) / worst unit.
Feasibility scan only — adoption requires the pre-registered rule +
LOUO stability protocol.  No test contact.
Output: v4_gate_redesign_results.txt, v4_gate_redesign.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_resid_clean import meandrop
from exp_v4_hi import SEEDS

LONG = {6, 8}
RES = os.path.join(HERE, 'v4_gate_redesign_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def detect_var(cur, dur, ucyc, sig='std'):
    """frozen detector with a selectable baseline noise estimator."""
    w = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
    mu0 = cur[:w].mean()
    if sig == 'std':
        sd0 = cur[:w].std(ddof=1) + 1e-8
    else:                                   # robust: scaled MAD
        med = np.median(cur[:w])
        sd0 = 1.4826 * np.median(np.abs(cur[:w] - med)) + 1e-8
    x = np.maximum(cur, mu0 - 10 * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def unit_data():
    """cache per (seed, unit): cc, ll, dc, ucyc, dur, true onset, nbase."""
    D = []
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                        if k.endswith('_cc')})
        for u in units:
            cc = Zd[f'u{u}_cc']; ll = Zd[f'u{u}_ll']; dc = Zd[f'u{u}_dc']
            true_on = int(Zd[f'u{u}_onset'][0])
            ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
            nbase = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
            D.append(dict(u=u, cc=cc, ll=ll, dc=dc, ucyc=ucyc, dur=dur,
                          true=true_on, nbase=nbase))
    return D


def curve(d, V):
    cur = np.empty(len(d['ucyc']))
    for i, c in enumerate(d['ucyc']):
        b = d['cc'] == c
        m = b if V is None else (b & (d['dc'] >= V))
        if not m.any():
            m = b
        cur[i] = np.percentile(d['ll'][m], 75)
    return cur


def score(D, vfun, sig='std'):
    errs = {'all': [], 'short': [], 'long': [], 'per': {}}
    for d in D:
        V = vfun(d)
        e = detect_var(curve(d, V), d['dur'], d['ucyc'], sig) - d['true']
        errs['all'].append(e)
        errs['short' if d['u'] not in LONG else 'long'].append(e)
        errs['per'].setdefault(d['u'], []).append(e)
    r = lambda a: float(np.sqrt((np.array(a, float) ** 2).mean()))
    worst = max(r(v) for v in errs['per'].values())
    wu = max(errs['per'], key=lambda u: r(errs['per'][u]))
    return r(errs['all']), r(errs['short']), r(errs['long']), worst, wu


def main():
    open(RES, 'w').close()
    t0 = time.time()
    D = unit_data()
    nb = sorted({(d['u'], d['nbase']) for d in D})
    log('baseline window cycle count per unit: ' +
        ' '.join(f'u{u}:{n}' for u, n in nb))
    log('')
    hdr = (f'  {"candidate":>34} | {"overall":>7} | {"short+med":>9} | '
           f'{"long":>6} | {"worst unit":>12}')

    log('REFERENCES')
    log(hdr)
    for tag, vf, sg in [('ungated', lambda d: None, 'std'),
                        ('frozen V*=20.2 std', lambda d: 20.2, 'std')]:
        o, s, l, w, wu = score(D, vf, sg)
        log(f'  {tag:>34} | {o:7.2f} | {s:9.2f} | {l:6.2f} | {w:8.2f} u{wu}')

    log('')
    log('A. COVERAGE-CONDITIONAL V  (nbase < 10 -> V_sparse, else V_dense)')
    log(hdr)
    outA = {}
    for vd in [19.9, 20.0, 20.1, 20.2]:
        for vs in [None, 18.5, 18.8, 19.0, 19.2, 19.5, 19.9]:
            vf = lambda d, vd=vd, vs=vs: (vs if d['nbase'] < 10 else vd)
            o, s, l, w, wu = score(D, vf, 'std')
            key = f'dense {vd:.1f} / sparse {"off" if vs is None else vs}'
            outA[key] = (o, s, l, w)
            log(f'  {key:>34} | {o:7.2f} | {s:9.2f} | {l:6.2f} | '
                f'{w:8.2f} u{wu}')

    log('')
    log('B. ROBUST SIGMA0 (scaled MAD) x single absolute V')
    log(hdr)
    outB = {}
    for V in [None, 18.8, 19.2, 19.5, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3,
              20.4, 20.5]:
        vf = lambda d, V=V: V
        o, s, l, w, wu = score(D, vf, 'mad')
        key = f'MAD, V={"off" if V is None else f"{V:.1f}"}'
        outB[key] = (o, s, l, w)
        log(f'  {key:>34} | {o:7.2f} | {s:9.2f} | {l:6.2f} | {w:8.2f} u{wu}')

    log('')
    log('C. COMBINED — best-feasible A cell with MAD sigma0')
    log(hdr)
    feasA = {k: v for k, v in outA.items() if v[2] <= 7.65 + 1e-9}
    bestA = min(feasA, key=lambda k: feasA[k][0]) if feasA else None
    if bestA:
        vd = float(bestA.split()[1]); vs_s = bestA.split()[-1]
        vs = None if vs_s == 'off' else float(vs_s)
        vf = lambda d: (vs if d['nbase'] < 10 else vd)
        o, s, l, w, wu = score(D, vf, 'mad')
        log(f'  {bestA + " + MAD":>34} | {o:7.2f} | {s:9.2f} | {l:6.2f} | '
            f'{w:8.2f} u{wu}')
    np.savez(os.path.join(HERE, 'v4_gate_redesign.npz'),
             **{f'A__{k}': np.array(v) for k, v in outA.items()},
             **{f'B__{k}': np.array(v) for k, v in outB.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
