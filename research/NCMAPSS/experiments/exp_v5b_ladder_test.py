"""
exp_v5b_ladder_test.py — the four-way onset scoring ladder evaluated on the
sealed TEST units = the TWELFTH DS03 opening, pre-registered, disclosed
(user instruction 2026-08-12: produce the ladder on test).

Rows (identical detector, per-cycle q75, conditional gate for MOGP only):
  1. zRMS      s = -(1/Q) sum_c (r_c/sigma_c)^2 ; sigma_c per model from its
               own DEV healthy-window (cycle<=3) residuals — frozen transfer
  2. fixedSig  s = -r' Sigma_hat^-1 r ; Sigma_hat per model from its own DEV
               healthy-window residual covariance — frozen transfer
  3. ownVar    s = -m2 = -r' S(w)^-1 r ; each model's own condition-dependent
               covariance (MOGP recovered from cached test dc/ll; competitors
               refit on the recorded 27k, validated against cached test ll)
  4. gaussLL   the official Gaussian log-likelihood (sanity: must reproduce
               8.54 / 9.01 / 6.87 / 10.03 / 8.88 / 10.66)
Report everything regardless of direction.  Test = 18 cases (6 units x 3
seeds), unit-level n = 6: descriptive, no significance claims.
Outputs: v5b_ladder_test_results.txt, v5b_ladder_test.npz
"""
import os, sys, time, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from exp_v4_bench import train_idx_27k
from exp_v4_bench_rul import sample_grid, SIDX, HP, MAKERS
from exp_v4_setcmp import detect
from exp_v4_hi import SEEDS

Q = 5
LOG2PI = math.log(2 * math.pi)
VD, VS, NB = 20.2, 19.0, 10
BENCH = ['llke', 'bspline', 'lr', 'cabn']
KINDS = ['zrms', 'fixed', 'm2', 'll']
RES = os.path.join(HERE, 'v5b_ladder_test_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def dev_healthy(sd, name):
    """model's own dev healthy-window residuals (frozen scoring constants)."""
    if name == 'mogp':
        Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                        if k.endswith('_cc')})
        h = np.concatenate([Zd[f'u{u}_resid'][Zd[f'u{u}_cc'] <= 3]
                            for u in units])
    else:
        Zr = np.load(os.path.join(HERE, f'v4_bench_resid_s{sd}.npz'))
        h = Zr[f'{name}_resid'][Zr['cc'] <= 3]
    sig = h.std(0) + 1e-12
    Sg = np.cov(h.T) + 1e-10 * np.eye(Q)
    return sig, np.linalg.inv(Sg)


def bench_test_m2(sd, cache):
    """refit competitors, validate test ll vs cache, return {name: m2_test}."""
    out_p = os.path.join(HERE, f'v5b_bench_test_m2_s{sd}.npz')
    if os.path.exists(out_p):
        return dict(np.load(out_p))
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= 3)[0]
    tr = train_idx_27k(sd, unit, cyc, pool)
    t_idx, t_cc, t_uu = sample_grid(At, sd)
    Zl = np.load(os.path.join(HERE, f'v4_bench_ll_s{sd}.npz'))
    assert np.array_equal(t_cc, Zl['t_cc']) and np.array_equal(t_uu, Zl['t_uu'])
    out = {}
    for name in BENCH:
        t0 = time.time()
        m = MAKERS[name](**HP[name])
        np.random.seed(0)
        m.fit(W[tr], X[tr])
        _, dcv, ll = m.stats(Wt[t_idx], Xt[t_idx], chunk=4096)
        ref = Zl[f'{name}_ll_test'].astype(np.float64)
        derr = float(np.median(np.abs(ll - ref)))
        assert derr < 1e-2, f'{name} s{sd}: test ll mismatch {derr}'
        out[f'{name}_m2'] = (2.0 * (dcv - ll) - Q * LOG2PI).astype(np.float32)
        log(f'  seed{sd} {name}: test refit+stats ok ({time.time()-t0:.0f}s, '
            f'll median dev {derr:.2e})')
    np.savez(out_p, **out)
    return out


def onset_errors(kind, name, gated, m2c):
    errs = []
    for sd in SEEDS:
        Zt = np.load(os.path.join(HERE, f'v4_test_stats_s{sd}.npz'))
        if name != 'mogp':
            Zl = np.load(os.path.join(HERE, f'v4_bench_ll_s{sd}.npz'))
            Zr = np.load(os.path.join(HERE, f'v4_bench_test_resid_s{sd}.npz'))
        if kind in ('zrms', 'fixed'):
            sig, Si = dev_healthy(sd, name)
        units = sorted({int(k[1:].split('_')[0]) for k in Zt.files
                        if k.endswith('_cc')})
        for u in units:
            if name == 'mogp':
                cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']
                dc = Zt[f'u{u}_dc']; ll = Zt[f'u{u}_ll']
                if kind == 'll':
                    s = ll
                elif kind == 'm2':
                    s = -(2.0 * (dc - ll) - Q * LOG2PI)
                elif kind == 'zrms':
                    s = -np.mean((rs / sig) ** 2, axis=1)
                else:
                    s = -np.einsum('ij,jk,ik->i', rs, Si, rs)
            else:
                mall = Zr['uu'] == u
                cc = Zr['cc'][mall]; rs = Zr[f'{name}_resid'][mall]; dc = None
                if kind == 'll':
                    s = Zl[f'{name}_ll_test'][Zl['t_uu'] == u]
                elif kind == 'm2':
                    s = -m2c[sd][f'{name}_m2'][Zl['t_uu'] == u]
                elif kind == 'zrms':
                    s = -np.mean((rs / sig) ** 2, axis=1)
                else:
                    s = -np.einsum('ij,jk,ik->i', rs, Si, rs)
            true_on = int(Zt[f'u{u}_onset'][0])
            ucyc = Zt[f'u{u}_ucyc']; dur = Zt[f'u{u}_hours'].astype(float)
            keep = np.isin(ucyc, np.unique(cc))
            ucyc, dur = ucyc[keep], dur[keep]
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
    log('V5B LADDER ON TEST — TWELFTH DS03 opening, pre-registered. '
        'Scoring constants frozen on dev; test = prediction input only. '
        'n = 6 units: descriptive.')
    cache = L.load_cache()
    m2c = {sd: bench_test_m2(sd, cache) for sd in SEEDS}
    rows = [('mogp cond gate', 'mogp', True), ('mogp ungated', 'mogp', False)]
    rows += [(n, n, False) for n in BENCH]
    r = lambda a: float(np.sqrt((a ** 2).mean()))
    out = {}
    log(f'\n  {"model":>16} | ' + ' | '.join(f'{k:>7}' for k in KINDS))
    for label, name, gated in rows:
        vals = []
        for kind in KINDS:
            e = onset_errors(kind, name, gated, m2c)
            out[f'{label.replace(" ", "_")}_{kind}'] = e
            vals.append(r(e))
        log(f'  {label:>16} | ' + ' | '.join(f'{v:7.2f}' for v in vals))
    log('\n  sanity: the ll column must reproduce '
        '8.54 / 9.01 / 6.87 / 10.03 / 8.88 / 10.66')
    np.savez(os.path.join(HERE, 'v5b_ladder_test.npz'), **out)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
