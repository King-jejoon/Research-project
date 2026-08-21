"""
exp_v4_vsweep_class.py — class-stratified re-scoring of the dev gate sweep.

Question (user, 2026-08-09): can a different absolute V alone remove the
long-haul gate penalty while keeping the short/medium gain?  Pure
re-analysis of cached point statistics (v4_c3_stats_s*.npz) — the same
sweep grid as the frozen chain (off + 18.1..20.5), scored on dev onset
RMSE overall AND per flight class (long = class 3: u6, u8; short/medium:
the other 7 units).  No GP refits, no test contact.  Candidate rule under
evaluation: overall minimum with tie band, subject to a long-haul
no-harm constraint (gated <= ungated on class 3).
Output: v4_vsweep_class_results.txt, v4_vsweep_class.npz
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_test import detect
from exp_v4_hi import SEEDS

LONG = {6, 8}                       # dev class-3 units
RES = os.path.join(HERE, 'v4_vsweep_class_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def onset_errs(V):
    """signed onset errors at gate V (V=None -> ungated), per unit tag."""
    errs, tags = [], []
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
            cur = np.empty(len(ucyc))
            for i, c in enumerate(ucyc):
                b = cc == c
                m = b if V is None else (b & (dc >= V))
                if not m.any():
                    m = b
                cur[i] = np.percentile(ll[m], 75)
            errs.append(detect(cur, dur, ucyc) - true_on)
            tags.append(u)
    return np.array(errs, float), np.array(tags)


def rmse(a):
    return float(np.sqrt((a ** 2).mean())) if len(a) else float('nan')


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('CLASS-STRATIFIED DEV GATE SWEEP — onset RMSE overall / short+medium '
        '/ long (u6, u8), cached stats, frozen detector')
    grid = [None] + list(np.round(np.arange(18.1, 20.51, 0.1), 1))
    out = {}
    e0, tg = onset_errs(None)
    long_m = np.isin(tg, list(LONG))
    ref = (rmse(e0), rmse(e0[~long_m]), rmse(e0[long_m]))
    log(f'  {"V":>5} | {"overall":>7} | {"short+med":>9} | {"long":>6} | '
        f'{"long gap vs off":>15}')
    log(f'  {"off":>5} | {ref[0]:7.2f} | {ref[1]:9.2f} | {ref[2]:6.2f} | '
        f'{"—":>15}')
    out['off'] = ref
    for V in grid[1:]:
        e, _ = onset_errs(V)
        o, s, l = rmse(e), rmse(e[~long_m]), rmse(e[long_m])
        out[f'{V:.1f}'] = (o, s, l)
        flag = ' <= off' if l <= ref[2] + 1e-9 else ''
        log(f'  {V:5.1f} | {o:7.2f} | {s:9.2f} | {l:6.2f} | '
            f'{l - ref[2]:+15.2f}{flag}')
    np.savez(os.path.join(HERE, 'v4_vsweep_class.npz'),
             **{k: np.array(v) for k, v in out.items()})
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
