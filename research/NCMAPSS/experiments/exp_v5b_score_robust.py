"""
exp_v5b_score_robust.py — distribution-assumption robustness check for the
onset benchmark (colleague question 2026-08-12: "the Gaussian assumption is
imposed — does it drive the comparison?").

Replace the per-point score: Gaussian log-likelihood  ->  NEGATIVE Mahalanobis
deviation  s = -m2,  m2 = (x-mu)' S^-1 (x-mu).  m2 uses only the first two
moments (mean + covariance) — no distributional family.  Everything else
byte-identical: per-cycle q75 curve, conditional gate for MOGP, frozen
detector (30 h baseline, clip 10 sigma, mean-drop), dev 27 cases.

m2 recovery: ll = detcov - m2/2 - (Q/2)log(2pi)  ->  m2 = 2(detcov - ll) - Q log(2pi).
  MOGP: exact from cached u_dc / u_ll.
  Competitors: detcov not cached -> refit each model on the recorded 27k rows
  (deterministic; seconds per fit), recompute (detcov, ll) on the recorded
  sampling grid and VALIDATE the recomputed ll against the cached ll before
  using m2.  DEV ONLY — no test contact.
Outputs: v5b_score_robust_results.txt, v5b_score_robust.npz
"""
import os, sys, time, math
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_bench2_lib as B2
from exp_v4_bench import train_idx_27k
from exp_v4_bench_rul import sample_grid, SIDX, HP, MAKERS
from exp_v4_setcmp import detect
from exp_v4_hi import SEEDS

Q = 5
LOG2PI = math.log(2 * math.pi)
VD, VS, NB = 20.2, 19.0, 10
BENCH = ['llke', 'bspline', 'lr', 'cabn']
RES = os.path.join(HERE, 'v5b_score_robust_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def bench_m2(sd, cache):
    """refit competitors, validate ll against cache, return {name: m2_dev}."""
    out_p = os.path.join(HERE, f'v5b_bench_m2_s{sd}.npz')
    if os.path.exists(out_p):
        return dict(np.load(out_p))
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= 3)[0]
    tr = train_idx_27k(sd, unit, cyc, pool)
    d_idx, d_cc, d_uu = sample_grid(A, sd)
    Zl = np.load(os.path.join(HERE, f'v4_bench_ll_s{sd}.npz'))
    assert np.array_equal(d_cc, Zl['d_cc']) and np.array_equal(d_uu, Zl['d_uu'])
    out = {}
    for name in BENCH:
        t0 = time.time()
        m = MAKERS[name](**HP[name])
        np.random.seed(0)
        m.fit(W[tr], X[tr])
        _, dcv, ll = m.stats(W[d_idx], X[d_idx], chunk=4096)
        ref = Zl[f'{name}_ll_dev'].astype(np.float64)
        derr = float(np.median(np.abs(ll - ref)))
        assert derr < 1e-2, f'{name} s{sd}: ll mismatch median {derr}'
        out[f'{name}_m2'] = (2.0 * (dcv - ll) - Q * LOG2PI).astype(np.float32)
        log(f'  seed{sd} {name}: refit+stats ok ({time.time()-t0:.0f}s, '
            f'll median dev {derr:.2e})')
    np.savez(out_p, **out)
    return out


def onset_rmse(kind, name, gated, m2_cache):
    """errors over dev 27 cases with score kind in {'ll', 'm2'}."""
    errs = []
    for sd in SEEDS:
        H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
        Zd = np.load(os.path.join(HERE, f'v4_c3_stats_s{sd}.npz'))
        if name != 'mogp':
            Zl = np.load(os.path.join(HERE, f'v4_bench_ll_s{sd}.npz'))
        units = sorted({int(k[1:].split('_')[0]) for k in Zd.files
                        if k.endswith('_cc')})
        for u in units:
            if name == 'mogp':
                cc = Zd[f'u{u}_cc']; dc = Zd[f'u{u}_dc']; ll = Zd[f'u{u}_ll']
                s = ll if kind == 'll' else -(2.0*(dc - ll) - Q*LOG2PI)
            else:
                mall = Zl['d_uu'] == u
                cc = Zl['d_cc'][mall]; dc = None
                s = (Zl[f'{name}_ll_dev'][mall] if kind == 'll'
                     else -m2_cache[sd][f'{name}_m2'][mall])
            true_on = int(Zd[f'u{u}_onset'][0])
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
    log('V5B SCORE ROBUSTNESS — onset benchmark rescored with the '
        'distribution-free deviation s = -Mahalanobis (dev only, 27 cases)')
    cache = L.load_cache()
    m2c = {sd: bench_m2(sd, cache) for sd in SEEDS}

    rows = [('mogp cond gate', 'mogp', True), ('mogp ungated', 'mogp', False)]
    rows += [(n, n, False) for n in BENCH]
    r = lambda a: float(np.sqrt((a ** 2).mean()))
    E = {}
    log(f'\n  {"model":>16} | {"ll score":>9} | {"-m2 score":>9} | '
        f'{"p vs cond (-m2)":>15}')
    for label, name, gated in rows:
        e_ll = onset_rmse('ll', name, gated, m2c)
        e_m2 = onset_rmse('m2', name, gated, m2c)
        E[label] = (e_ll, e_m2)
    cond_m2 = E['mogp cond gate'][1]
    out = {}
    for label, (e_ll, e_m2) in E.items():
        p = ('    —' if label == 'mogp cond gate' else
             f'{st.wilcoxon(np.abs(cond_m2), np.abs(e_m2), zero_method="zsplit").pvalue:.4f}')
        log(f'  {label:>16} | {r(e_ll):9.2f} | {r(e_m2):9.2f} | {p:>15}')
        out[f'{label.replace(" ", "_")}_ll'] = e_ll
        out[f'{label.replace(" ", "_")}_m2'] = e_m2
    log('\n  sanity: the ll column must reproduce '
        '5.21 / 6.78 / 7.67 / 8.69 / 10.02 / 10.54')
    np.savez(os.path.join(HERE, 'v5b_score_robust.npz'), **out)
    log(f'\nwall={(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
