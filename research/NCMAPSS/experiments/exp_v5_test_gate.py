"""
exp_v5_test_gate.py — ONE-TIME test verification of the v5 coverage-
conditional gate (8th DS03 test opening, user-authorized, disclosed).

Rule frozen on dev BEFORE this run (exp_v4_gate_redesign + stability
battery): baseline-window cycle count < 10 -> V_sparse = 19.0,
else V_dense = V* = 20.2.  Detector otherwise byte-identical to the
frozen chain (q75 LL, 30 flight-h baseline, std sigma0, clip 10 sigma,
mean-drop).  No GP refits: pure re-summary of v4_test_stats_s*.npz.

Pre-registered report: (1) sanity gates — frozen-gate 11.11 and ungated
9.01 must reproduce from the same cache; (2) test onset overall /
long (u10, u11, u13) / short+med (u12, u14, u15) / per unit, for
ungated / frozen / v5; (3) mechanism — per-unit sigma0 ratio vs ungated
and the kept-vs-rejected LL gap in baseline cycles (flip incidence).
Output: v5_test_gate_results.txt, v5_test_gate.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_test import detect
from exp_v4_hi import SEEDS

VD, VS, NB = 20.2, 19.0, 10
LONG = {10, 11, 13}
RES = os.path.join(HERE, 'v5_test_gate_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def load_units(sd):
    Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                    if k.endswith('_cc')})
    D = []
    for u in units:
        cc = Zt[f'u{u}_cc']; ll = Zt[f'u{u}_ll']; dc = Zt[f'u{u}_dc']
        ucyc = Zt[f'u{u}_ucyc']; dur = Zt[f'u{u}_hours'].astype(float)
        keep = np.isin(ucyc, np.unique(cc))
        ucyc, dur = ucyc[keep], dur[keep]
        nb = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
        D.append(dict(u=u, cc=cc, ll=ll, dc=dc, ucyc=ucyc, dur=dur,
                      true=int(Zt[f'u{u}_onset'][0]), nb=nb))
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


def vrule(d, mode):
    if mode == 'ungated':
        return None
    if mode == 'frozen':
        return VD
    return VS if d['nb'] < NB else VD          # v5


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5 COVERAGE-CONDITIONAL GATE — one-time test verification '
        '(8th DS03 test opening, disclosed).  Rule frozen on dev: '
        f'nbase<{NB} -> V={VS}, else V={VD}.')
    E = {m: {} for m in ['ungated', 'frozen', 'v5']}
    S0 = {m: {} for m in ['ungated', 'frozen', 'v5']}
    nbs = {}
    for sd in SEEDS:
        for d in load_units(sd):
            nbs[d['u']] = d['nb']
            for m in E:
                cur = curve(d, vrule(d, m))
                e = detect(cur, d['dur'], d['ucyc']) - d['true']
                E[m].setdefault(d['u'], []).append(e)
                w = max(5, int(np.searchsorted(np.cumsum(d['dur']), 30.0) + 1))
                S0[m].setdefault(d['u'], []).append(
                    float(cur[:w].std(ddof=1)))
    units = sorted(E['v5'])
    log('baseline cycle count: ' + ' '.join(f'u{u}:{nbs[u]}' for u in units)
        + f'  -> sparse rule fires on: '
        + ' '.join(f'u{u}' for u in units if nbs[u] < NB))

    r = lambda a: float(np.sqrt((np.array(a, float) ** 2).mean()))
    log('')
    log('SANITY GATES (must reproduce the benchmark table):')
    all_f = [e for u in units for e in E['frozen'][u]]
    all_u = [e for u in units for e in E['ungated'][u]]
    log(f'  frozen gate overall = {r(all_f):.2f}  (expected 11.11)')
    log(f'  ungated overall     = {r(all_u):.2f}  (expected 9.01)')

    log('')
    log(f'  {"variant":>8} | {"overall":>7} | {"long u10/11/13":>14} | '
        f'{"short+med":>9} | {"mean delay":>10}')
    out = {}
    for m in ['ungated', 'frozen', 'v5']:
        alle = [e for u in units for e in E[m][u]]
        lng = [e for u in units if u in LONG for e in E[m][u]]
        sht = [e for u in units if u not in LONG for e in E[m][u]]
        out[m] = (r(alle), r(lng), r(sht), float(np.mean(alle)))
        log(f'  {m:>8} | {r(alle):7.2f} | {r(lng):14.2f} | {r(sht):9.2f} | '
            f'{np.mean(alle):+10.2f}')

    log('')
    log('per-unit signed onset error, 3 seeds  (ungated | frozen | v5):')
    for u in units:
        log(f'  u{u} nb={nbs[u]:>2}: '
            + ' '.join(f'{e:+3.0f}' for e in E['ungated'][u]) + '  |  '
            + ' '.join(f'{e:+3.0f}' for e in E['frozen'][u]) + '  |  '
            + ' '.join(f'{e:+3.0f}' for e in E['v5'][u]))

    log('')
    log('mechanism — sigma0 ratio vs ungated (mean over seeds) and '
        'kept-vs-rejected LL gap in baseline cycles at the applied V:')
    for u in units:
        rf = np.mean(np.array(S0['frozen'][u]) / np.array(S0['ungated'][u]))
        rv = np.mean(np.array(S0['v5'][u]) / np.array(S0['ungated'][u]))
        # flip check on seed 0
        d = [x for x in load_units(0) if x['u'] == u][0]
        w = max(5, int(np.searchsorted(np.cumsum(d['dur']), 30.0) + 1))
        gaps = {}
        for m, V in [('frozen', VD), ('v5', vrule(d, 'v5'))]:
            gs = []
            for c in d['ucyc'][:w]:
                b = d['cc'] == c
                if V is None:
                    continue
                k = b & (d['dc'] >= V); rj = b & ~(d['dc'] >= V)
                if k.any() and rj.any():
                    gs.append(float(d['ll'][k].mean() - d['ll'][rj].mean()))
            gaps[m] = min(gs) if gs else float('nan')
        log(f'  u{u}: sigma0 frozen {rf:5.2f}x  v5 {rv:5.2f}x   '
            f'worst kept-rejected LL gap: frozen {gaps["frozen"]:+7.2f}  '
            f'v5 {gaps["v5"]:+7.2f}')

    np.savez(os.path.join(HERE, 'v5_test_gate.npz'),
             **{f'{m}_u{u}': np.array(E[m][u]) for m in E for u in E[m]})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
