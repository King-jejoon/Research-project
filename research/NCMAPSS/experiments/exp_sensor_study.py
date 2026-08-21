"""
exp_sensor_study.py — sensor-count study: does adding sensors reduce onset RMSE?

Sets: 3=[T48,T50,Wf]  5=[T30,T48,T50,Nc,Wf]  7=[T24,T30,T48,T50,Ps30,Nc,Wf]
For each set, the same frozen methodology:
  A. kernel CV      : {rbf, matern52, matern32, matern12} x rank1 x 3 seeds,
                      shared train/held-out sets (rng 1000+sd), held-out zRMSE
  B. final models   : stage-1 GP -> candidate detcov -> drop lowest 5%
                      -> resample 8192 -> retrain   (rng 2000+sd / 3000+sd)
  C. unit stats     : dev 9 units, every cycle, 200 pts: per-point LL (fixed
                      whitening) + detcov  -> cached npz
  D. detector sweep : same grid as the 3-sensor study (cycle stat mean/median/
                      trim25/q75, detcov gate, EWMA w/wo detrend) -> best RMSE
Usage: --set 5|7 [--probe]      (3-sensor results already exist: RMSE 10.17)
"""
import os, sys, time, itertools, argparse
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2

DT = torch.float32
SETS = {3: ['T48', 'T50', 'Wf'],
        5: ['T30', 'T48', 'T50', 'Nc', 'Wf'],
        7: ['T24', 'T30', 'T48', 'T50', 'Ps30', 'Nc', 'Wf']}
KERNELS = ['rbf', 'matern52', 'matern32', 'matern12']
SEEDS = [0, 1, 2]
NPTS, NCAND, NEVAL = 8192, 16384, 8192
M, STEPS, LR, MCOND, NPER, TRAIN_CYC, CLEAN = 18, 120, 0.1, 15, 200, 5, 0.05

ap = argparse.ArgumentParser()
ap.add_argument('--set', type=int, choices=[5, 7])
ap.add_argument('--sensors', type=str, default=None,
                help='comma list overriding --set, e.g. T50,T48,Wf,Nc,T30,P40,Ps30')
ap.add_argument('--tag', type=str, default=None)
ap.add_argument('--probe', action='store_true')
args = ap.parse_args()
if args.sensors:
    SENS = args.sensors.split(',')
    K = len(SENS)
    TAG = args.tag or f'{K}c'
else:
    K = args.set
    SENS = SETS[K]
    TAG = args.tag or str(K)
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
RES = os.path.join(HERE, f'sensor{TAG}_study_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit(Wtr, Ytr, kernel):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT)
    tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=K, rank=1, kernel=kernel).to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=K).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=M, num_steps=STEPS, lr=LR,
                           group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


@torch.no_grad()
def stats_of(gp, Wq, Xq=None, chunk=4096):
    """returns (pred_raw, detcov, llfix or None)"""
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = None if Xq is None else torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, m2, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                               gp['tY'], gp['struct'], teX,
                                               m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf if Xq is not None else np.zeros(len(dc)))
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def main():
    open(RES, 'a').close()
    log(f'=== SENSOR-{K} STUDY  {SENS}  ({time.strftime("%H:%M")}) ===')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit_a = A[:, 0].astype(int); cyc_a = A[:, 1].astype(int); hs_a = A[:, 3]
    pool = np.where(cyc_a < TRAIN_CYC)[0]
    t00 = time.time()

    if args.probe:
        rng = np.random.default_rng(1000)
        tr = pool[rng.permutation(len(pool))[:NPTS]]
        t0 = time.time()
        gp = fit(W[tr], X[tr], 'rbf')
        t_train = time.time() - t0
        t0 = time.time()
        stats_of(gp, W[tr[:4096]], X[tr[:4096]])
        t_stats = time.time() - t0
        log(f'[probe] train={t_train:.0f}s  stats4096={t_stats:.0f}s -> '
            f'CV~{t_train*4*3/60:.0f}min  models~{t_train*6/60:.0f}min  '
            f'unitstats~{t_stats*132000/4096*3/60:.0f}min')
        return

    # ---- A. kernel CV (rank1) ----
    cv_path = os.path.join(HERE, f'sensor{TAG}_cv.npz')
    if os.path.exists(cv_path):
        zz = np.load(cv_path); Z = {k: list(zz[k]) for k in zz.files}
    else:
        Z = {}
        for sd in SEEDS:
            rng = np.random.default_rng(1000 + sd)
            perm = rng.permutation(len(pool))
            tr = pool[perm[:NPTS]]; ev = pool[perm[NPTS:NPTS + NEVAL]]
            yscale = StandardScaler().fit(X[tr]).scale_
            for kernel in KERNELS:
                t0 = time.time()
                try:
                    gp = fit(W[tr], X[tr], kernel)
                    pred, _, _ = stats_of(gp, W[ev])
                    z = float(np.sqrt((((X[ev] - pred) / yscale) ** 2).mean()))
                except Exception as e:
                    log(f'  CV seed{sd} {kernel}: FAILED {type(e).__name__}'); z = np.nan
                Z.setdefault(kernel, []).append(z)
                log(f'  CV seed{sd} {kernel:>9}: zRMSE={z:.4f} ({time.time()-t0:.0f}s)')
        np.savez(cv_path, **{k: np.array(v) for k, v in Z.items()})
    best_k = min(Z, key=lambda k: np.nanmean(Z[k]))
    log(f'[A] kernel ranking: ' + '  '.join(f'{k}={np.nanmean(Z[k]):.4f}' for k in
        sorted(Z, key=lambda k: np.nanmean(Z[k]))) + f'  -> WINNER {best_k}')

    # ---- B+C. final models + unit stats ----
    for sd in SEEDS:
        sp = os.path.join(HERE, f'sensor{TAG}_stats_s{sd}.npz')
        if os.path.exists(sp):
            continue
        t0 = time.time()
        rng = np.random.default_rng(2000 + sd)
        cand = pool[rng.permutation(len(pool))[:NCAND]]
        tr1 = cand[rng.choice(NCAND, NPTS, replace=False)]
        gp1 = fit(W[tr1], X[tr1], best_k)
        _, dc, _ = stats_of(gp1, W[cand])
        keep = cand[dc >= np.percentile(dc, CLEAN * 100)]
        rng2 = np.random.default_rng(3000 + sd)
        tr = keep[rng2.choice(len(keep), NPTS, replace=False)]
        gp = fit(W[tr], X[tr], best_k)
        log(f'[B] seed{sd}: final model ({time.time()-t0:.0f}s)')
        out = {}
        for u in np.unique(unit_a):
            rows = np.where(unit_a == u)[0]
            cyc_u = cyc_a[rows]; hs_u = hs_a[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ucyc = np.unique(cyc_u)
            idx, cc = [], []
            for c in ucyc:
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                idx.append(r); cc.append(np.full(len(r), c))
            idx = np.concatenate(idx); cc = np.concatenate(cc)
            _, dcv, ll = stats_of(gp, W[idx], X[idx])
            hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
            below = np.where(hs_by < 0.5)[0]
            onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
            out[f'u{u}_cc'] = cc.astype(np.int32)
            out[f'u{u}_ll'] = ll.astype(np.float32)
            out[f'u{u}_dc'] = dcv.astype(np.float32)
            out[f'u{u}_onset'] = np.array([onset])
        np.savez_compressed(sp, **out)
        log(f'[C] seed{sd}: unit stats cached ({time.time()-t0:.0f}s total)')

    # ---- D. detector sweep (same grid as 3-sensor study) ----
    S = {}
    for sd in SEEDS:
        z = np.load(os.path.join(HERE, f'sensor{TAG}_stats_s{sd}.npz'))
        units = sorted({int(kk[1:].split('_')[0]) for kk in z.files})
        S[sd] = (z, units, np.concatenate([z[f'u{u}_dc'] for u in units]))
    STATS = {'mean': np.mean, 'median': np.median,
             'trim25': lambda v: np.mean(np.sort(v)[int(.25 * len(v)):int(.75 * len(v))]),
             'q75': lambda v: np.percentile(v, 75)}

    def curves(sd, q, stat):
        z, units, dcp = S[sd]
        V = -np.inf if q == 0 else np.percentile(dcp, q)
        f = STATS[stat]; out = []
        for u in units:
            cc, ll, dc = z[f'u{u}_cc'], z[f'u{u}_ll'], z[f'u{u}_dc']
            onset = int(z[f'u{u}_onset'][0]); ucyc = np.unique(cc)
            cur = np.array([f(ll[(cc == c) & (dc >= V)]) if ((cc == c) & (dc >= V)).any()
                            else f(ll[cc == c]) for c in ucyc])
            out.append((ucyc, cur, onset))
        return out

    def ewma_det(cur, w, lam, kk, consec, detrend):
        idx = np.arange(w, dtype=float)
        if detrend:
            Amat = np.vstack([idx, np.ones(w)]).T
            beta = np.linalg.lstsq(Amat, cur[:w], rcond=None)[0]
            x = cur - (np.arange(len(cur)) * beta[0] + beta[1])
        else:
            x = cur - cur[:w].mean()
        sd0 = x[:w].std(ddof=1) + 1e-8
        lcl = -kk * sd0 * np.sqrt(lam / (2 - lam)); e = 0.0; run = 0
        for i, v in enumerate(x):
            e = lam * v + (1 - lam) * e
            if i >= w:
                run = run + 1 if e < lcl else 0
                if run >= consec: return i - consec + 1
        return len(cur)

    CUR = {}
    for stat in STATS:
        for q in [0, 5, 10, 15, 20]:
            for sd in SEEDS:
                CUR[(stat, q, sd)] = curves(sd, q, stat)
    res = []
    for stat, q, w, lam, kk, consec, dt in itertools.product(
            STATS.keys(), [0, 5, 10, 15, 20], [10, 12, 15],
            [0.1, 0.2, 0.3], [2.5, 3.0, 4.0], [1, 2], [False, True]):
        ds = []
        for sd in SEEDS:
            for ucyc, cur, onset in CUR[(stat, q, sd)]:
                kdet = ewma_det(cur, w, lam, kk, consec, dt)
                det = int(ucyc[kdet]) if kdet < len(ucyc) else int(ucyc[-1] + 1)
                ds.append(det - onset)
        d = np.array(ds, float)
        res.append((float(np.sqrt((d ** 2).mean())), stat, q, w, lam, kk, consec, dt, d))
    res.sort(key=lambda r: r[0])
    log(f'[D] sensor-{K} TOP 5:')
    for r in res[:5]:
        log(f'  RMSE={r[0]:5.2f}  stat={r[1]} q={r[2]} warm={r[3]} lam={r[4]} '
            f'kk={r[5]} consec={r[6]} detrend={r[7]}')
    best = res[0]
    d = best[8]
    for sd in SEEDS:
        units = S[sd][1]
        log(f'  seed{sd}: ' + ' '.join(f'u{u}{int(v):+d}' for u, v in
                                       zip(units, d[sd * 9:(sd + 1) * 9])))
    log(f'[DONE] sensor-{K}: best onset RMSE = {best[0]:.2f}  '
        f'(3-sensor reference: 10.17)   wall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
