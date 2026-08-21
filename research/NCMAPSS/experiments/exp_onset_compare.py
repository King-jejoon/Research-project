"""
exp_onset_compare.py — fault-onset detection accuracy: baseline GPR vs
detcov-5%-cleaned GPR (training-data cleaning), DS03, ground truth = hs.

Models (rebuilt exactly as in exp_traindata_detcov.py, same seeds/rng order):
  A. baseline : plain 8192 pts from the 16384 candidate pool (기존 방식)
  B. cleaned  : candidates' lowest-5%-detcov removed -> resample 8192 -> retrain

Detection (identical for both models, no test-time filtering):
  per unit: 200 pts/cycle (seed = unit*7+sd, port convention)
  -> per-point fixed-whitening conditional LL (demo_cond2)
  -> per-cycle mean LL -> mean-drop changepoint (search range [5, n-5))
  -> detected onset cycle
Ground truth: first cycle whose mean hs < 0.5 (dev AND test units of DS03).
Metrics per unit: signed delay, |delay|, per-cycle label accuracy.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from demo_cond2 import conditional_stats2
from exp_traindata_detcov import (fit_gp, detcov_of, SENS, SIDX, NPTS, NCAND,
                                  NEVAL, TRAIN_CYC, true_onset_mask, DT)

SEEDS = [0, 1, 2]
NPER = 200
CLEAN_DROP = 0.05
RES = os.path.join(HERE, 'onset_compare_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def mean_drop_changepoint(x, kmin=5):
    x = np.asarray(x, float); n = len(x)
    kmax = max(kmin + 1, n - 5)
    sigma = np.std(x, ddof=1) + 1e-12
    cs = np.cumsum(x)
    T = np.full(n - 1, -np.inf)
    for k in range(kmin, kmax):
        m1 = cs[k - 1] / k
        m2 = (cs[-1] - cs[k - 1]) / (n - k)
        T[k - 1] = np.sqrt(k * (n - k) / n) * (m1 - m2) / sigma
    return int(np.argmax(T) + 1)


def build_models(sd, W, X, pool):
    """Exact same rng call order as exp_traindata_detcov.main()."""
    rng = np.random.default_rng(sd)
    perm = rng.permutation(len(pool))
    cand = pool[perm[:NCAND]]
    _evin = pool[perm[NCAND:NCAND + NEVAL]]
    # (evgn draw consumed from rng in the sweep; replicate the call)
    healthy_gen = np.where(true_onset_mask(L.load_cache()['A_dev']))[0]
    _evgn = rng.choice(healthy_gen, NEVAL, replace=False)

    base_tr = cand[rng.choice(NCAND, NPTS, replace=False)]
    t0 = time.time()
    gpA = fit_gp(W[base_tr], X[base_tr])
    dc_cand = detcov_of(gpA, W[cand])
    thr = np.percentile(dc_cand, CLEAN_DROP * 100)
    keep = cand[dc_cand >= thr]
    rng_d = np.random.default_rng(sd * 100 + int(CLEAN_DROP * 100))
    tr = keep[rng_d.choice(len(keep), NPTS, replace=False)]
    gpB = fit_gp(W[tr], X[tr])
    log(f'seed{sd}: models built (A=baseline, B=clean{CLEAN_DROP:.0%})  '
        f'({time.time()-t0:.0f}s)')
    return gpA, gpB


@torch.no_grad()
def unit_ll_curve(gp, Wu, Xu, cyc_u, sd, u):
    """per-cycle mean fixed-LL for one unit."""
    rng = np.random.default_rng(int(u) * 7 + sd)
    ucyc = np.unique(cyc_u)
    idx_all, cc_all = [], []
    for c in ucyc:
        r = np.where(cyc_u == c)[0]
        if len(r) > NPER:
            r = rng.choice(r, NPER, replace=False)
        idx_all.append(r); cc_all.append(np.full(len(r), c))
    idx = np.concatenate(idx_all); cc = np.concatenate(cc_all)
    ll = np.empty(len(idx), np.float64)
    for i in range(0, len(idx), 4096):
        sl = idx[i:i + 4096]
        teX = torch.tensor(gp['xs'].transform(Wu[sl]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xu[sl]), dtype=DT)
        _, _, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                             gp['tY'], gp['struct'], teX,
                                             m=15, test_y=teY)
        ll[i:i + 4096] = llf
    return ucyc, np.array([ll[cc == c].mean() for c in ucyc])


def true_onset(hs_u, cyc_u, ucyc):
    hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
    below = np.where(hs_by < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def main():
    open(RES, 'w').close()
    log(f'ONSET DETECTION: baseline GPR vs detcov-{CLEAN_DROP:.0%}-cleaned GPR '
        f'(DS03, truth=hs, seeds={SEEDS}, NPER={NPER}, cp on fixed-LL)')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    Wt, Xt, At = cache['W_test'], cache['X_s_test'][:, SIDX], cache['A_test']
    pool = np.where(A[:, 1].astype(int) < TRAIN_CYC)[0]

    recs = {'A': [], 'B': []}
    t00 = time.time()
    for sd in SEEDS:
        gpA, gpB = build_models(sd, W, X, pool)
        for split, (Wd, Xd, Ad) in [('dev', (W, X, A)), ('test', (Wt, Xt, At))]:
            units = np.unique(Ad[:, 0].astype(int))
            for u in units:
                rows = np.where(Ad[:, 0].astype(int) == u)[0]
                cyc_u = Ad[rows, 1].astype(int); hs_u = Ad[rows, 3]
                Wu, Xu = Wd[rows], Xd[rows]
                for name, gp in [('A', gpA), ('B', gpB)]:
                    ucyc, llc = unit_ll_curve(gp, Wu, Xu, cyc_u, sd, u)
                    tru = true_onset(hs_u, cyc_u, ucyc)
                    k = mean_drop_changepoint(llc)
                    det = int(ucyc[k]) if k < len(ucyc) else int(ucyc[-1] + 1)
                    acc = float(((ucyc >= det) == (ucyc >= tru)).mean())
                    recs[name].append(dict(sd=sd, split=split, unit=int(u),
                                           true=tru, det=det,
                                           delay=det - tru, acc=acc))
            log(f'  seed{sd} {split} done ({time.time()-t00:.0f}s)')

    # ---- report ----
    log('')
    log(f'{"model":>18} | {"mean delay":>10} | {"mean |delay|":>12} | '
        f'{"label acc":>9} | {"within ±3cyc":>12}')
    log('-' * 74)
    for name, lbl in [('A', 'baseline GPR'), ('B', f'detcov {CLEAN_DROP:.0%} clean')]:
        rr = recs[name]
        dl = np.array([r['delay'] for r in rr]); ac = np.array([r['acc'] for r in rr])
        log(f'{lbl:>18} | {dl.mean():>+10.2f} | {np.abs(dl).mean():>12.2f} | '
            f'{100*ac.mean():>8.1f}% | {100*(np.abs(dl)<=3).mean():>11.1f}%')
    for split in ['dev', 'test']:
        log(f'\n[{split}] per-unit detected onset (true | A base | B clean), seed-mean:')
        units = sorted({r['unit'] for r in recs['A'] if r['split'] == split})
        for u in units:
            ra = [r for r in recs['A'] if r['split'] == split and r['unit'] == u]
            rb = [r for r in recs['B'] if r['split'] == split and r['unit'] == u]
            log(f'  u{u:>2}: true={ra[0]["true"]:>3} | A={np.mean([r["det"] for r in ra]):>5.1f} '
                f'(d{np.mean([r["delay"] for r in ra]):+5.1f}) | '
                f'B={np.mean([r["det"] for r in rb]):>5.1f} '
                f'(d{np.mean([r["delay"] for r in rb]):+5.1f})')
    np.savez(os.path.join(HERE, 'onset_compare.npz'),
             A=np.array([(r['sd'], r['unit'], r['true'], r['det']) for r in recs['A']]),
             B=np.array([(r['sd'], r['unit'], r['true'], r['det']) for r in recs['B']]))
    log(f'\nwall = {(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
