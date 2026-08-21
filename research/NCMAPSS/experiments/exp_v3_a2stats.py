"""
exp_v3_a2stats.py — STAGE A2 (part 1): per-point residual statistics for the
per-set winning model (rbf rank1 for every set, from the Stage-A grid).

SET env 3|7 (set5 reuses v2_stats_s{seed}.npz — identical configuration and
identical stratified draw).  Per seed: rebuild the winner model on the shared
split (cycle<3, 1000/(unit,cycle) = 18k, same rng as the grid), then for every
dev unit and cycle sample 200 rows (rng unit*7+seed) and cache per-point
residual / detcov / corrected LL and the hs onset.
Output: v3_stats_set{SET}_s{seed}.npz
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 5)))
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
        7: ['T24', 'T30', 'T48', 'T50', 'Ps30', 'Nc', 'Wf']}
SET = int(os.environ.get('SET', 3))
SENS = SETS[SET]
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
TRAIN_CYC, NPER_TRAIN, NPER = 3, 1000, 200
M, STEPS, LR, MCOND = 18, 120, 0.1, 15
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, f'v3_a2stats_set{SET}_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit(Wtr, Ytr, tries=4):
    for t in range(tries):
        xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
        tX = torch.tensor(xs.transform(Wtr), dtype=DT)
        tY = torch.tensor(ys.transform(Ytr), dtype=DT)
        model = GPyTorchMOGP(4, num_tasks=len(SENS), rank=1, kernel='rbf').to('cpu', DT)
        lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=len(SENS)).to('cpu', DT)
        try:
            r = train_vecchia_mogp(model, lik, tX, tY, m=M, num_steps=STEPS,
                                   lr=LR, group=True, verbose=False)
            model.eval(); lik.eval()
            return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure,
                        xs=xs, ys=ys), t
        except torch._C._LinAlgError:
            # jitter the draw: drop 50 random rows, duplicate-safe redraw below
            rg = np.random.default_rng(1234 + t)
            keep = rg.permutation(len(Wtr))[:len(Wtr) - 50]
            Wtr, Ytr = Wtr[keep], Ytr[keep]
    raise RuntimeError('fit failed after retries')


@torch.no_grad()
def point_stats(gp, Wq, Xq, chunk=4096):
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                              gp['tY'], gp['struct'], teX,
                                              m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'A2 STATS — set{SET} {SENS}, rbf rank1, cycle<3 stratified 18k')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int); hs = A[:, 3]
    pool = np.where(cyc < TRAIN_CYC)[0]

    for sd in SEEDS:
        out_path = os.path.join(HERE, f'v3_stats_set{SET}_s{sd}.npz')
        if os.path.exists(out_path):
            log(f'seed{sd}: cached'); continue
        rng = np.random.default_rng(sd)
        tr = []
        for u in np.unique(unit[pool]):
            for c in range(1, TRAIN_CYC):
                rows = pool[(unit[pool] == u) & (cyc[pool] == c)]
                tr.append(rng.choice(rows, min(NPER_TRAIN, len(rows)), replace=False))
        tr = np.concatenate(tr)
        t0 = time.time()
        gp, nret = fit(W[tr], X[tr])
        log(f'seed{sd}: fitted ({time.time()-t0:.0f}s, retries={nret})')
        out = {}
        for u in np.unique(unit):
            rows = np.where(unit == u)[0]
            cyc_u = cyc[rows]; hs_u = hs[rows]
            rng_u = np.random.default_rng(int(u) * 7 + sd)
            ucyc = np.unique(cyc_u)
            idx, cc = [], []
            for c in ucyc:
                r = rows[cyc_u == c]
                if len(r) > NPER:
                    r = rng_u.choice(r, NPER, replace=False)
                idx.append(r); cc.append(np.full(len(r), c))
            idx = np.concatenate(idx); cc = np.concatenate(cc)
            pred, dcv, ll = point_stats(gp, W[idx], X[idx])
            hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
            below = np.where(hs_by < 0.5)[0]
            onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
            out[f'u{u}_cc'] = cc.astype(np.int32)
            out[f'u{u}_resid'] = (X[idx] - pred).astype(np.float32)
            out[f'u{u}_dc'] = dcv.astype(np.float32)
            out[f'u{u}_ll'] = ll.astype(np.float32)
            out[f'u{u}_onset'] = np.array([onset])
        np.savez(out_path, **out)
        log(f'seed{sd}: stats cached ({time.time()-t0:.0f}s total)')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
