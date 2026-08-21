"""
exp_v4_grid_backfill.py — v4 STAGE 2b: uniform retry ladder for grid cells
that ended in a Cholesky failure, so the final table has a number in every
cell (defense requirement: no missing values).

Policy (identical for every configuration, by construction):
  attempt 1 : the original grid fit (exp_v4_grid.py) — cells that succeeded
              there are NOT touched; their numbers are final
  attempt 2 : refit with cholesky jitter 1e-5
  attempt 3 : refit with cholesky jitter 1e-4
  attempt 4 : drop 50 training rows (rng 1234) and refit with jitter 1e-4
Each rescued cell is annotated with the attempt that converged (j1/j2/d3 in
the table footnote); selection rule is unchanged (min mean zRMSE, within-sd
ties -> simpler config) and is re-run on the completed table.

SET env 3|5|7.  Reads v4_grid_set{SET}.npz, writes
v4_grid_set{SET}_backfilled.npz and appends to v4_grid_set{SET}_results.txt.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 4)))
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2
from exp_v4_grid import SETS, KERNELS, SEEDS, M, STEPS, LR, MCOND, \
    TRAIN_CYC_MAX, NPER_TRAIN, NEVAL, DT, zrmse

SET = int(os.environ.get('SET', 3))
SENS = SETS[SET]
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
RES = os.path.join(HERE, f'v4_grid_set{SET}_results.txt')
NPZ_IN = os.path.join(HERE, f'v4_grid_set{SET}.npz')
NPZ_OUT = os.path.join(HERE, f'v4_grid_set{SET}_backfilled.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


PLANS = [('j1', 1e-5, 0, torch.float32), ('j2', 1e-4, 0, torch.float32),
         ('d3', 1e-4, 50, torch.float32), ('j3', 1e-3, 0, torch.float32),
         ('d4', 1e-3, 50, torch.float32), ('f5', 1e-5, 0, torch.float64)]
SKIP_TO = os.environ.get('SKIP_TO', 'j1')   # resume ladder from this rung


def fit_ladder(Wtr, Ytr, kernel, rank):
    """attempts 2..7 of the uniform policy; returns (gp, tag)."""
    start = [i for i, p in enumerate(PLANS) if p[0] == SKIP_TO][0]
    for tag, jit, ndrop, dt in PLANS[start:]:
        Wc, Yc = Wtr, Ytr
        if ndrop:
            keep = np.random.default_rng(1234).permutation(len(Wtr))[:len(Wtr) - ndrop]
            Wc, Yc = Wtr[keep], Ytr[keep]
        xs = StandardScaler().fit(Wc); ys = StandardScaler().fit(Yc)
        tX = torch.tensor(xs.transform(Wc), dtype=dt)
        tY = torch.tensor(ys.transform(Yc), dtype=dt)
        model = GPyTorchMOGP(4, num_tasks=len(SENS), rank=rank,
                             kernel=kernel).to('cpu', dt)
        lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=len(SENS)).to('cpu', dt)
        try:
            kw = ({'float_value': jit} if dt == torch.float32
                  else {'double_value': jit})
            with gpytorch.settings.cholesky_jitter(**kw):
                r = train_vecchia_mogp(model, lik, tX, tY, m=M,
                                       num_steps=STEPS, lr=LR,
                                       group=True, verbose=False)
            model.eval(); lik.eval()
            return dict(model=model, lik=lik, tX=tX, tY=tY,
                        struct=r.structure, xs=xs, ys=ys), tag
        except torch._C._LinAlgError:
            continue
    return None, 'xx'


@torch.no_grad()
def zrmse_dt(gp, Wq, Xq, chunk=4096):
    """held-out zRMSE matching the gp's dtype (float32 or float64)."""
    dt = gp['tX'].dtype
    errs = []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=dt)
        pred_s, *_ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                        gp['tY'], gp['struct'], teX,
                                        m=MCOND, test_y=None)
        errs.append(pred_s - gp['ys'].transform(Xq[i:i + chunk]))
    e = np.concatenate(errs)
    return float(np.sqrt((e ** 2).mean()))


def main():
    t00 = time.time()
    log('')
    log(f'STAGE 2b BACKFILL — set{SET}: retry ladder on Cholesky-failed cells')
    src = NPZ_OUT if os.path.exists(NPZ_OUT) else NPZ_IN
    log(f'  (source table: {os.path.basename(src)}, ladder from {SKIP_TO})')
    Z = dict(np.load(src, allow_pickle=True))
    prev_tags = {}
    if 'tags' in Z:
        for s in Z.pop('tags'):
            key, tg = str(s).split(':')
            prev_tags[key] = tg.split(',')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int)
    pool = np.where(cyc <= TRAIN_CYC_MAX)[0]
    splits = {}
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        tr = []
        for u in np.unique(unit[pool]):
            for c in range(1, TRAIN_CYC_MAX + 1):
                rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
                tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)),
                                     replace=False))
        tr = np.concatenate(tr)
        rest = np.setdiff1d(pool, tr, assume_unique=False)
        ev = rng.choice(rest, NEVAL, replace=False)
        splits[sd] = (tr, ev)

    tags = {}
    for k in KERNELS:
        for r in [1, 2, 3]:
            key = f'{k}_r{r}'
            arr = Z[key].astype(float)
            tg = list(prev_tags.get(key, ['--'] * len(SEEDS)))
            for sd in SEEDS:
                if not np.isnan(arr[sd]):
                    if tg[sd] in ('--', 'xx'):
                        tg[sd] = 'a1'
                    continue
                t0 = time.time()
                tr, ev = splits[sd]
                gp, tag = fit_ladder(W[tr], X[tr], k, r)
                if gp is None:
                    log(f'  {k:>9} r{r} seed{sd}: STILL FAILS after ladder '
                        f'({time.time()-t0:.0f}s)')
                else:
                    arr[sd] = zrmse_dt(gp, W[ev], X[ev])
                    log(f'  {k:>9} r{r} seed{sd}: rescued[{tag}] '
                        f'zRMSE={arr[sd]:.5f}  ({time.time()-t0:.0f}s)')
                tg[sd] = tag
            Z[key] = arr
            tags[key] = tg

    log('')
    log('completed table (mean zRMSE over available seeds; tags per seed: '
        'a1=first-try, j1/j2=jitter 1e-5/1e-4, d3=50-row redraw):')
    log(f'  {"kernel":>9} | ' + ' | '.join(f'{"rank "+str(r):>26}'
                                           for r in [1, 2, 3]))
    means = {}
    for k in KERNELS:
        cells = []
        for r in [1, 2, 3]:
            arr = Z[f'{k}_r{r}']; ok = arr[~np.isnan(arr)]
            means[(k, r)] = np.mean(ok) if len(ok) else np.inf
            tg = ','.join(tags[f'{k}_r{r}'])
            cells.append(f'{np.mean(ok):.5f}±{np.std(ok):.5f}[{tg}]'
                         if len(ok) else f'FAIL[{tg}]')
        log(f'  {k:>9} | ' + ' | '.join(f'{c:>26}' for c in cells))

    # eligibility: only configurations with ALL seeds converged (under the
    # uniform ladder) may compete — a 1-seed mean is not a comparable number
    complete = {c: m for c, m in means.items()
                if not np.isnan(Z[f'{c[0]}_r{c[1]}']).any()}
    excl = sorted(c for c in means if c not in complete)
    best = min(complete, key=complete.get)
    sd_best = np.nanstd(Z[f'{best[0]}_r{best[1]}'])
    simple_order = {k: i for i, k in enumerate(KERNELS)}
    cands = [c for c in complete if complete[c] - complete[best] <= sd_best]
    sel = sorted(cands, key=lambda c: (simple_order[c[0]], c[1]))[0]
    log('')
    if excl:
        log(f'excluded from selection (unconverged seeds remain, '
            f'reported with footnote only): {excl}')
    log(f'best by mean (complete configs): {best[0]} r{best[1]} '
        f'({complete[best]:.5f}); within-sd candidates: {sorted(cands)}')
    log(f'RESELECTED (simplicity rule, complete configs): '
        f'kernel={sel[0]}, rank={sel[1]}')
    np.savez(NPZ_OUT, **{k: v for k, v in Z.items()
                         if k not in ('winner_kernel', 'winner_rank')},
             tags=np.array([f'{k}:{",".join(tags[k])}' for k in tags]),
             winner_kernel=np.array([sel[0]]), winner_rank=np.array([sel[1]]))
    log(f'backfill wall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
