"""
exp_v5b_zrms_onset.py — onset detection with the user's proposed zRMS
compression: each residual channel divided by its own healthy-window noise
sigma (per-channel z-scoring, DIAGONAL — no cross-channel correlation),
combined as RMS.  Score per point: s = -(1/Q) * sum_c (r_c / sigma_c)^2.
Identical procedure for every model (sigma_c from its own cycle<=3
residuals) — fully symmetric in the colleague's sense.  Everything else
frozen (q75, conditional gate for MOGP, 30 h / clip 10 sigma / mean-drop).
DEV ONLY, 27 cases.
Outputs: v5b_zrms_onset_results.txt
"""
import os, sys, time
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_v4_setcmp import detect
from exp_v4_hi import SEEDS

VD, VS, NB = 20.2, 19.0, 10
BENCH = ['llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'v5b_zrms_onset_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def unit_arrays(sd, name):
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                    if k.endswith('_cc')})
    data, healthy = {}, []
    if name == 'mogp':
        for u in units:
            cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
            data[u] = (cc, rs, dc, int(Zd[f'u{u}_onset'][0]))
            healthy.append(rs[cc <= 3])
    else:
        Zr = np.load(os.path.join(HERE, f'v4_bench_resid_s{sd}.npz'))
        cc_all, uu_all, rs_all = Zr['cc'], Zr['uu'], Zr[f'{name}_resid']
        for u in units:
            m = uu_all == u
            data[u] = (cc_all[m], rs_all[m], None, int(Zd[f'u{u}_onset'][0]))
            healthy.append(rs_all[m][cc_all[m] <= 3])
    sig = np.concatenate(healthy).std(0) + 1e-12          # per-channel sigma
    return H, units, data, sig


def onset_errors(name, gated):
    errs = []
    for sd in SEEDS:
        H, units, data, sig = unit_arrays(sd, name)
        for u in units:
            cc, rs, dc, true_on = data[u]
            s = -np.mean((rs / sig) ** 2, axis=1)          # -zRMS^2
            ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            dur = np.array([durs[pos[int(c)]] for c in ucyc], float)
            if gated:
                nb = max(5, int(np.searchsorted(np.cumsum(dur), 30.0) + 1))
                V = VS if nb < NB else VD
            cur = np.empty(len(ucyc))
            for i, c in enumerate(ucyc):
                b = cc == c
                mm = (b & (dc >= V)) if (gated and dc is not None) else b
                if not mm.any():
                    mm = b
                cur[i] = np.percentile(s[mm], 75)
            errs.append(detect(cur, dur, ucyc) - true_on)
    return np.array(errs, float)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('V5B ZRMS ONSET — per-channel healthy-noise z-scoring (diagonal), '
        'RMS combination; identical procedure for every model.  Dev 27 cases.')
    rows = [('mogp cond gate', 'mogp', True), ('mogp ungated', 'mogp', False)]
    rows += [(n, n, False) for n in BENCH]
    r = lambda a: float(np.sqrt((a ** 2).mean()))
    E = {label: onset_errors(name, gated) for label, name, gated in rows}
    cond = E['mogp cond gate']
    log(f'\n  {"model":>16} | {"zRMS RMSE":>9} | {"p vs cond":>10}')
    for label, e in E.items():
        p = ('     —' if label == 'mogp cond gate' else
             f'{st.wilcoxon(np.abs(cond), np.abs(e), zero_method="zsplit").pvalue:.4f}')
        log(f'  {label:>16} | {r(e):9.2f} | {p:>10}')
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
