"""
exp_v4_bench.py — v4 NORMAL-MODEL benchmark, model level only (user decision:
the old residual-pipeline benchmark is discarded).

Question: on the exact data the chain's GP was trained on, how do the
competing conditional models compare AS PREDICTORS?

  data      the v4 GPR training set itself: dev 9 units, cycle<=3, stratified
            1,000 rows per (unit, cycle) = 27,000 rows per seed (identical
            draw to the frozen chain); nothing outside these rows is used
  protocol  5-fold cross-validation x 3 seeds (15 fold-evaluations per model);
            folds are random partitions of the 27,000 rows, shared by every
            model within a (seed, fold)
  models    mogp   Vecchia MOGP, RBF rank 1, 5 sensors (the chain's model,
                   frozen internals: m=18, Adam 0.1 x 120, mcond=15)
            lr / bspline / llke / cabn   from exp_bench2_lib (same predictive
                   definitions: pred, detcov, ll)
  hp        selected per (seed, fold) on an inner 80/20 split of the training
            folds by mean predictive log-likelihood (proper scoring rule),
            then refit on the full training folds
  metrics   held-out zRMSE (model's own standardised space, grid convention),
            per-sensor raw RMSE, mean predictive log-likelihood,
            training / inference wall time
Outputs: v4_bench_results.txt, v4_bench_s{sd}_f{f}.npz (resumable per fold)
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from exp_v4_c3_detcov import fit as mogp_fit, point_stats, SIDX, SENS
import exp_bench2_lib as B2

SEEDS = [0, 1, 2]
NFOLD = 5
TRAIN_CYC_MAX, NPER_TRAIN = 3, 1000
GRIDS = {
    'lr':      [dict()],
    'bspline': [dict(n_knots=k) for k in (3, 5, 8, 12, 20, 30)],
    'llke':    [dict(h=h) for h in (0.05, 0.1, 0.15, 0.2, 0.35, 0.5)],
    'cabn':    [dict(lam1=l) for l in (0.0, 0.001, 0.01, 0.05)],
}
MAKERS = {'lr': B2.LinearNM, 'bspline': B2.BSplineNM,
          'llke': B2.LLKENM, 'cabn': B2.CaBNNM}
RES = os.path.join(HERE, 'v4_bench_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def train_idx_27k(sd, unit, cyc, pool):
    rng = np.random.default_rng(sd)
    tr = []
    for u in np.unique(unit[pool]):
        for c in range(1, TRAIN_CYC_MAX + 1):
            rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
            tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)),
                                 replace=False))
    return np.concatenate(tr)


def metrics_from(pred, ll, ys, X_te):
    zr = float(np.sqrt(((ys.transform(pred) - ys.transform(X_te)) ** 2)
                       .mean()))
    per = np.sqrt(((pred - X_te) ** 2).mean(0))
    return zr, per, float(np.mean(ll))


def run_fold(sd, f, W, X, tr, te):
    out_path = os.path.join(HERE, f'v4_bench_s{sd}_f{f}.npz')
    if os.path.exists(out_path):
        log(f'seed{sd} fold{f}: cached')
        return
    res = {}
    # inner split of the training folds for hp selection
    rng = np.random.default_rng(200 + 10 * sd + f)
    perm = rng.permutation(len(tr))
    n_in = int(0.8 * len(tr))
    itr, ival = tr[perm[:n_in]], tr[perm[n_in:]]

    for name, maker in MAKERS.items():
        best, best_ll = None, -np.inf
        for hp in GRIDS[name]:
            if len(GRIDS[name]) == 1:
                best = hp
                break
            m = maker(**hp)
            np.random.seed(0)
            m.fit(W[itr], X[itr])
            _, _, ll = m.stats(W[ival], X[ival])
            v = float(np.mean(ll))
            if v > best_ll:
                best_ll, best = v, hp
        m = maker(**best)
        np.random.seed(0)
        t0 = time.time(); m.fit(W[tr], X[tr]); t_fit = time.time() - t0
        t0 = time.time(); pred, _, ll = m.stats(W[te], X[te])
        t_inf = time.time() - t0
        zr, per, mll = metrics_from(pred, ll, m.ys, X[te])
        res[name] = dict(zrmse=zr, per=per, ll=mll, tfit=t_fit, tinf=t_inf,
                         hp=str(best))
        log(f'seed{sd} fold{f} {name:>8}: zRMSE={zr:.5f}  LL={mll:7.3f}  '
            f'fit={t_fit:6.1f}s inf={t_inf:5.1f}s  hp={best}')

    t0 = time.time(); gp, nret = mogp_fit(W[tr], X[tr])
    t_fit = time.time() - t0
    t0 = time.time(); pred, _, ll = point_stats(gp, W[te], X[te])
    t_inf = time.time() - t0
    zr, per, mll = metrics_from(pred, ll, gp['ys'], X[te])
    res['mogp'] = dict(zrmse=zr, per=per, ll=mll, tfit=t_fit, tinf=t_inf,
                       hp=f'retries={nret}')
    log(f'seed{sd} fold{f}     mogp: zRMSE={zr:.5f}  LL={mll:7.3f}  '
        f'fit={t_fit:6.1f}s inf={t_inf:5.1f}s')

    np.savez(out_path, **{f'{n}_{k}': v for n, d in res.items()
                          for k, v in d.items() if k != 'hp'},
             **{f'{n}_hp': np.array([d['hp']]) for n, d in res.items()})


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'V4 BENCHMARK — model level, 5-fold CV x 3 seeds on the 27k GPR '
        f'training data (sensors {SENS})')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= TRAIN_CYC_MAX)[0]

    for sd in SEEDS:
        t27 = train_idx_27k(sd, unit, cyc, pool)
        assert len(t27) == 27000
        folds = np.array_split(np.random.default_rng(100 + sd)
                               .permutation(len(t27)), NFOLD)
        for f in range(NFOLD):
            te = t27[folds[f]]
            tr = t27[np.concatenate([folds[g] for g in range(NFOLD)
                                     if g != f])]
            run_fold(sd, f, W, X, tr, te)

    # ---------------- aggregate ----------------
    names = ['mogp', 'llke', 'bspline', 'lr', 'cabn']
    agg = {n: {k: [] for k in ['zrmse', 'll', 'tfit', 'tinf', 'per']}
           for n in names}
    for sd in SEEDS:
        for f in range(NFOLD):
            Z = np.load(os.path.join(HERE, f'v4_bench_s{sd}_f{f}.npz'))
            for n in names:
                for k in ['zrmse', 'll', 'tfit', 'tinf', 'per']:
                    agg[n][k].append(Z[f'{n}_{k}'])
    log('')
    log('SUMMARY — 15 fold-evaluations (5-fold CV x 3 seeds), mean ± sd')
    log(f'  {"model":>8} | {"zRMSE":>17} | {"pred LL":>16} | '
        f'{"fit s":>8} | {"infer s":>8}')
    for n in names:
        zr = np.array(agg[n]['zrmse']); llv = np.array(agg[n]['ll'])
        log(f'  {n:>8} | {zr.mean():.5f} ± {zr.std():.5f} | '
            f'{llv.mean():8.3f} ± {llv.std():5.3f} | '
            f'{np.mean(agg[n]["tfit"]):8.1f} | {np.mean(agg[n]["tinf"]):8.2f}')
    log('')
    log('per-sensor raw RMSE (mean over 15 folds):')
    log(f'  {"model":>8} | ' + ' | '.join(f'{s:>8}' for s in SENS))
    for n in names:
        per = np.mean(np.stack(agg[n]['per']), 0)
        log(f'  {n:>8} | ' + ' | '.join(f'{v:8.4f}' for v in per))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
