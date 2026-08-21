"""
exp_v5b_fixed_sigma.py — fully symmetric onset scoring (colleague fairness
question, method 1): EVERY model's covariance estimated by the IDENTICAL
procedure — the fixed covariance of its own healthy-window (cycle<=3) fit
residuals.  The GP's condition-dependent covariance is switched off, so
"the variance comes from fitting errors" holds for all models alike; the
remaining difference is pure predictive-mean quality.

Score per point: s = -(x-mu)' Sigma_hat^-1 (x-mu)   (model's own Sigma_hat)
Everything else identical: per-cycle q75, conditional gate for MOGP (plus
an ungated MOGP row), frozen detector, dev 27 cases.  DEV ONLY.
Outputs: v5b_fixed_sigma_results.txt, v5b_fixed_sigma.npz
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
RES = os.path.join(HERE, 'v5b_fixed_sigma_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def unit_arrays(sd, name):
    """per unit: cc, resid, dc(or None); plus pooled healthy resid rows."""
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
            cc = cc_all[m]; rs = rs_all[m]
            data[u] = (cc, rs, None, int(Zd[f'u{u}_onset'][0]))
            healthy.append(rs[cc <= 3])
    Sg = np.cov(np.concatenate(healthy).T) + 1e-10 * np.eye(5)
    Si = np.linalg.inv(Sg)
    return H, units, data, Si


def onset_errors(name, gated):
    errs = []
    for sd in SEEDS:
        H, units, data, Si = unit_arrays(sd, name)
        for u in units:
            cc, rs, dc, true_on = data[u]
            s = -np.einsum('ij,jk,ik->i', rs, Si, rs)
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
    log('V5B FIXED-SIGMA — fully symmetric scoring: every model, own fixed '
        'healthy-window residual covariance only (GP condition-dependent '
        'covariance OFF).  Dev 27 cases.')
    rows = [('mogp cond gate', 'mogp', True), ('mogp ungated', 'mogp', False)]
    rows += [(n, n, False) for n in BENCH]
    r = lambda a: float(np.sqrt((a ** 2).mean()))
    E = {label: onset_errors(name, gated) for label, name, gated in rows}
    cond = E['mogp cond gate']
    out = {}
    log(f'\n  {"model":>16} | {"fixed-Sigma RMSE":>16} | {"p vs cond":>10}')
    for label, e in E.items():
        p = ('     —' if label == 'mogp cond gate' else
             f'{st.wilcoxon(np.abs(cond), np.abs(e), zero_method="zsplit").pvalue:.4f}')
        log(f'  {label:>16} | {r(e):16.2f} | {p:>10}')
        out[label.replace(' ', '_')] = e
    np.savez(os.path.join(HERE, 'v5b_fixed_sigma.npz'), **out)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
