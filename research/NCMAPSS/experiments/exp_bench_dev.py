"""exp_bench_dev.py — DEV-ONLY benchmark of HI-construction methods.

Design A: the residual table that feeds the HI network (`_trim` of
rul_input_s{sd}.npz) is the shared input; only the HI-construction block is
swapped.  The first-passage stage (exponential basis, beta grid, truncation
fractions) is the frozen one imported from exp_rul_r23.

Protocol (DS03 dev 9 units only, test never touched here):
  * leave-one-unit-out: the HI model is REFITTED on the 8 training units in
    every fold -- for every method including ours -- so no method sees the
    held-out unit while building its HI.  (The official pipeline fits the HI net
    on all 9 dev units, so the `ours` row here can differ slightly from 7.36.)
  * beta chosen per method on the pooled dev LOO NASA score, same criterion as
    the frozen pipeline.
  * method hyper-parameters chosen on the same dev LOO NASA score, then frozen
    for the single test pass (exp_bench_test.py).

Outputs: bench_dev_results.txt, bench_dev_sel.npz
"""
import os, sys, time, json
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_bench_hi as B
from exp_rul_r23 import eval_units, nasa, FRACS

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'bench_dev_results.txt')

# widened beta grid: the frozen pipeline uses [16..50], but the baselines pushed
# against the lower edge, so the grid is extended downwards for EVERY method.
BETA_C = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 45, 50]

# `smooth` = length of a trailing moving average on the HI (0 = none), offered to
# the baselines only; the constrained MLP is smooth by construction.
SM = (0, 3, 5, 7)
# lam1 must reach down to 0: with lam1 >= 0.05 and a short healthy sample the
# NOTEARS solution collapses to W = 0, i.e. CaBN degenerates to a diagonal
# Gaussian and stops being a Bayesian network at all.
CONFIGS = {
    'ours':    [dict()],
    'lr':      [dict(smooth=s) for s in SM],
    'bspline': [dict(n_knots=k, smooth=s) for k in (0, 1, 2, 3) for s in SM],
    'llke':    [dict(h=h, smooth=s) for h in (1.0, 1.5, 2.5, 4.0) for s in SM],
    'cabn':    [dict(n_healthy=nh, lam1=l, stat='sqrt', smooth=s)
                for nh in (5, 10, 15) for l in (0.0, 0.001, 0.005, 0.01, 0.05)
                for s in SM],
}


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def loo_curves(name, hp, sd, units, R, cov):
    """refit per fold -> HI curves; returns folds, timings."""
    folds, tf, tp = [], [], []
    for u in units:
        tr = [v for v in units if v != u]
        m = B.build(name, **hp)
        hi_tr = m.fit_all(R, cov, tr, seed=sd)
        t0 = time.perf_counter()
        hi_te = m.predict(R[u], cov[u])
        tp.append(time.perf_counter() - t0)
        tf.append(m.t_fit)
        folds.append((u, tr, hi_tr, hi_te))
    return folds, float(np.mean(tf)), float(np.mean(tp))


def score_beta(folds, beta):
    P, T, F = [], [], []
    for u, tr, hi_tr, hi_te in folds:
        ctr = {v: np.arange(len(hi_tr[v])) / 500.0 for v in tr}
        cte = {u: np.arange(len(hi_te)) / 500.0}
        p, t, f, _ = eval_units(beta, ctr, hi_tr, cte, {u: hi_te}, FRACS, None)
        P.append(p); T.append(t); F.append(f)
    return np.concatenate(P), np.concatenate(T), np.concatenate(F)


def main():
    open(RES, 'w').close()
    log('BENCHMARK (design A): HI-construction methods on the frozen residual table')
    log(f'  input = rul_input_s*.npz `_trim` (5 sensors), dev 9 units, seeds={SEEDS}')
    log(f'  LOO with per-fold HI refit for every method; beta grid={BETA_C}; '
        f'fracs={FRACS}\n')
    t00 = time.time()
    selected = {}

    for name, cfgs in CONFIGS.items():
        rows = []
        for hp in cfgs:
            rm, sc, bt, tf, tp, err = [], [], [], [], [], []
            per_frac = {f: [] for f in FRACS}
            for sd in SEEDS:
                units, R, cov = B.load_split(sd, 'dev')
                folds, t_fit, t_pred = loo_curves(name, hp, sd, units, R, cov)
                best = None
                for b in BETA_C:
                    P, T, F = score_beta(folds, b)
                    s = nasa(P, T)
                    if best is None or s < best[0]:
                        best = (s, b, P, T, F)
                s, b, P, T, F = best
                rm.append(float(np.sqrt(((P - T) ** 2).mean())))
                sc.append(s); bt.append(b); tf.append(t_fit); tp.append(t_pred)
                err.append(P - T)
                for f in FRACS:
                    m = F == f
                    per_frac[f].append(float(np.sqrt(((P[m] - T[m]) ** 2).mean())))
            rows.append(dict(hp=hp, err=np.concatenate(err),
                             rmse=float(np.mean(rm)), rmse_sd=float(np.std(rm)),
                             nasa=float(np.mean(sc)), beta=bt,
                             t_fit=float(np.mean(tf)), t_pred=float(np.mean(tp)),
                             per_frac={f: float(np.mean(v)) for f, v in per_frac.items()}))
            if name == 'cabn':
                mm = B.build(name, **hp)
                units, R, cov = B.load_split(SEEDS[0], 'dev')
                mm.fit(R, cov, units)
                rows[-1]['nnz'] = int((mm.W != 0).sum())
            log(f'  {name:>8} {str(hp):<44} RMSE={rows[-1]["rmse"]:6.2f} '
                f'NASA={rows[-1]["nasa"]:8.1f} beta={bt} '
                f'fit={rows[-1]["t_fit"]:.2f}s pred={rows[-1]["t_pred"]*1e3:.1f}ms '
                + (f'nnz(W)={rows[-1]["nnz"]} ' if 'nnz' in rows[-1] else '')
                + f'({time.time()-t00:.0f}s)')
        # CaBN must keep a non-empty DAG, otherwise the "Bayesian network" part
        # is gone and the baseline is no longer Wei et al.'s method.  Configs
        # with nnz(W) = 0 are reported above but excluded from the selection.
        pool = [r for r in rows if r.get('nnz', 1) > 0] or rows
        sel = min(pool, key=lambda r: r['nasa'])
        selected[name] = sel
        log(f'  -> {name} selected {sel["hp"]}\n')

    log('')
    log(f'{"method":>8} | {"RMSE":>12} | {"NASA":>8} | {"20%":>6} {"40%":>6} '
        f'{"60%":>6} {"80%":>6} | {"train(s)":>8} {"test(ms)":>8}')
    log('-' * 92)
    for name, s in selected.items():
        pf = s['per_frac']
        log(f'{name:>8} | {s["rmse"]:6.2f}+/-{s["rmse_sd"]:4.2f} | {s["nasa"]:8.1f} | '
            f'{pf[0.2]:6.2f} {pf[0.4]:6.2f} {pf[0.6]:6.2f} {pf[0.8]:6.2f} | '
            f'{s["t_fit"]:8.2f} {s["t_pred"]*1e3:8.1f}')

    log('')
    log('paired comparison against `ours` over the 108 dev LOO cases '
        '(9 units x 4 truncations x 3 seeds), absolute error:')
    log(f'{"method":>8} | {"MAE":>6} | {"ours better":>11} | {"Wilcoxon p":>10}')
    log('-' * 48)
    e0 = np.abs(selected['ours']['err'])
    for name, s in selected.items():
        e1 = np.abs(s['err'])
        if name == 'ours':
            log(f'{name:>8} | {e0.mean():6.2f} | {"-":>11} | {"-":>10}')
            continue
        win = float((e0 < e1).mean())
        p = float(wilcoxon(e0, e1).pvalue)
        log(f'{name:>8} | {e1.mean():6.2f} | {win:>10.0%} | {p:>10.2e}')

    for v in selected.values():
        v.pop('err', None)
    np.savez(os.path.join(HERE, 'bench_dev_sel.npz'),
             sel=np.array([json.dumps(selected)], dtype=object), allow_pickle=True)
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
