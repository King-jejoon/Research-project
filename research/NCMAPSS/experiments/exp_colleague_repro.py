"""
exp_colleague_repro.py — RUN THE COLLEAGUE'S DEMO NOTEBOOK AS-IS on DS02.

Faithful reproduction of /Users/a1/Desktop/master/DEMO/Permutation&Grouping-MOGP.ipynb
(cells 0-19) as a CPU script.  Everything the notebook does is kept, including:

  * inputs = df_W.iloc[:, 1:]  ->  [Mach, TRA, T2, unit-id]  (alt dropped,
    unit id injected — exactly what the notebook's indexing does)
  * outputs = X_s columns [1,2,3,12,13] = [T24? see X_s_var] (5 sensors)
  * train  = cycle<5 rows of ALL dev units, subsampled by DEMO/train_idx.npy
  * MOGP   = GPyTorchMOGP(input_dim=4, rank=1, matern32), m=100, 200 steps,
             lr=0.01, MultitaskGaussianLikelihood
  * per-cycle conditional stats on one unit: 120 random points / cycle,
    m=200, grouped test subset, ORIGINAL quadratic  A = L22 @ r  (the
    un-whitened version, vecchia_mmd.py:435) — bug kept on purpose
  * LL     = detcov + quadratic - 0.5*log(2*pi)      (notebook cell 12)
  * filter = keep points with detcov > 3.60          (notebook cell 13)
  * changepoint = mean_drop_changepoint, k in [5,30) (notebook cell 15)

Only unavoidable deviations (documented):
  d1. device = cpu (no CUDA on this machine), torch threads = 6
  d2. np.random.choice seeded (--seed) so the run is repeatable
  d3. only the dev split of the h5 is loaded (test split unused by notebook)
  d4. the notebook's `change_cycle` line indexes the FULL df_A with a
      position from the unit subset (cell 14).  We report BOTH that faithful
      value and the correctly computed onset cycle.

Usage:
  probe (timing only):   exp_colleague_repro.py --probe
  full run:              exp_colleague_repro.py --unit 10
Artifacts: colleague_repro_model.pt, colleague_repro_u<unit>.npz,
           colleague_repro_u<unit>_results.txt, fig_colleague_repro_u<unit>_*.png
"""
import os, sys, time, math, argparse, pickle
import numpy as np
import h5py
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))  # M3 Max: 10 P-cores
import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

DEMO = __import__('exp_paths').DEMO
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEMO)

from vecchia_mmd import (
    mmd_ordering, _test_neighbours, _automatic_grouping_subset,
    _effective_group_neighbours, flatten_multioutput_covariance,
    _stabilize_covariance,
)
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp

H5 = os.path.join(__import__('exp_paths').DATA, 'N-CMAPSS_DS02-006.h5')
DT = torch.float32
DEV = torch.device('cpu')                       # d1
OUTPUT_IDX = np.array([1, 2, 3, 12, 13])        # notebook cell 8
DETCOV_THR = 3.60                               # notebook cell 13
Q = len(OUTPUT_IDX)


def log(s, res):
    print(s, flush=True)
    with open(res, 'a') as f:
        f.write(s + '\n')


def load_dev():
    with h5py.File(H5, 'r') as hdf:
        W = np.array(hdf.get('W_dev'))
        Xs = np.array(hdf.get('X_s_dev'))
        A = np.array(hdf.get('A_dev'))
        W_var = list(np.array(np.array(hdf.get('W_var')), dtype='U20'))
        Xs_var = list(np.array(np.array(hdf.get('X_s_var')), dtype='U20'))
        A_var = list(np.array(np.array(hdf.get('A_var')), dtype='U20'))
    return W, Xs, A, W_var, Xs_var, A_var


def make_features(W, A, fix_inputs):
    """faithful: notebook iloc[:,1:] quirk -> [Mach, TRA, T2, unit-id]
    fixed:    all four operating conditions  -> [alt, Mach, TRA, T2]"""
    if fix_inputs:
        return W[:, :4].copy()
    return np.column_stack([W[:, 1], W[:, 2], W[:, 3], A[:, 0]])


def build_training(W, Xs, A, fix_inputs):
    """Notebook cells 7-8 (train pool + train_idx subsample)."""
    unit = A[:, 0]
    cycle = A[:, 1]
    pool = []
    for u in np.unique(unit):
        pool.append(np.where((cycle < 5) & (unit == u))[0])
    pool = np.hstack(pool)
    train_idx = np.load(os.path.join(DEMO, 'train_idx.npy'))
    rows = pool[train_idx]
    Xfeat = make_features(W, A, fix_inputs)
    train_X_array = Xfeat[rows]
    train_Y_array = Xs[rows][:, OUTPUT_IDX]
    return train_X_array, train_Y_array


def fit_or_load(train_X_array, train_Y_array, steps, lr, m, res, force=False,
                ckpt_name='colleague_repro_model.pt'):
    ckpt = os.path.join(HERE, ckpt_name)
    xs = StandardScaler().fit(train_X_array)
    ys = StandardScaler().fit(train_Y_array)
    tX = torch.tensor(xs.transform(train_X_array), dtype=DT, device=DEV)
    tY = torch.tensor(ys.transform(train_Y_array), dtype=DT, device=DEV)
    model = GPyTorchMOGP(input_dim=4, num_tasks=Q, rank=1, kernel='matern32').to(DEV, DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Q).to(DEV, DT)
    if os.path.exists(ckpt) and not force:
        blob = torch.load(ckpt, weights_only=False)
        model.load_state_dict(blob['model'])
        lik.load_state_dict(blob['lik'])
        structure = blob['structure']
        log(f'[train] loaded checkpoint {ckpt} (steps={blob["steps"]})', res)
    else:
        t0 = time.time()
        r = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=lr,
                               group=True, verbose=True)
        structure = r.structure
        log(f'[train] m={m} steps={steps} lr={lr}: final LL={r.log_likelihoods[-1]:.1f} '
            f'({time.time()-t0:.0f}s)', res)
        torch.save(dict(model=model.state_dict(), lik=lik.state_dict(),
                        structure=structure, steps=steps), ckpt)
    model.eval(); lik.eval()
    return model, lik, tX, tY, structure, xs, ys


@torch.no_grad()
def cycle_stats(model, lik, tX, tY, structure, teX, teY, m=200, jitter=1e-6):
    """Notebook cell 12 body for one cycle — ORIGINAL quadratic kept."""
    n_train = tX.shape[0]; n_test = teX.shape[0]
    mu_train_ord = model.mean(tX).index_select(0, structure.order)
    x_train_ord = structure.ordered(tX)
    y_train_ord = structure.ordered(tY)

    test_order = mmd_ordering(teX)
    x_test_ord = teX.index_select(0, test_order)
    y_test_ord = teY.index_select(0, test_order)
    mu_test_ord = model.mean(teX).index_select(0, test_order)

    x_all = torch.cat([x_train_ord, x_test_ord], 0)
    mu_all = torch.cat([mu_train_ord, mu_test_ord], 0)
    values_all = torch.empty(n_train + n_test, Q, dtype=DT, device=DEV)
    values_all[:n_train] = y_train_ord
    values_all[n_train:] = mu_test_ord

    test_neighbours = _test_neighbours(x_all, n_train=n_train, n_test=n_test, m=m)
    neighbours = structure.neighbours + test_neighbours
    groups = _automatic_grouping_subset(neighbours, range(n_train, n_train + n_test), m=m)
    eff = _effective_group_neighbours(neighbours, groups, device=x_all.device)
    covfn = lambda a, b: model.observation_covariance(lik, a, b)

    quadratic, detcov, residual = [], [], []
    for i in range(n_train, n_train + n_test):
        ji = eff[i]; cond = ji[ji != i]
        if cond.numel() == 0:
            values_all[i] = mu_all[i]
            continue
        query = torch.cat([cond, torch.tensor([i], dtype=torch.long, device=DEV)])
        k = flatten_multioutput_covariance(covfn(x_all.index_select(0, query),
                                                 x_all.index_select(0, query)))
        cs = int(cond.numel()) * Q
        k = _stabilize_covariance(k, jitter=jitter)
        centered = (values_all.index_select(0, cond) - mu_all.index_select(0, cond)).reshape(-1, 1)
        chol = torch.linalg.cholesky(k)
        L11 = chol[:cs, :cs]; L21 = chol[cs:, :cs]; L22 = chol[cs:, cs:]
        alpha = torch.linalg.solve_triangular(L11, centered, upper=False)
        values_all[i] = mu_all[i] + (L21 @ alpha).reshape(Q)
        mu = values_all[i] - y_test_ord[i - n_train]
        A = L22 @ mu                                  # ORIGINAL (un-whitened)
        quadratic.append(-0.5 * (A.T @ A).reshape(-1))
        detcov.append(-1.0 * torch.log(torch.diagonal(L22)).sum())
        residual.append(mu)

    quadratic = torch.stack(quadratic).reshape(-1)
    detcov = torch.stack(detcov).reshape(-1)
    residual = torch.stack(residual)
    pred_ordered = values_all[n_train:]

    inv = torch.empty_like(test_order)
    inv[test_order] = torch.arange(n_test, device=DEV)
    return (pred_ordered.index_select(0, inv).cpu().numpy(),
            quadratic.index_select(0, inv).cpu().numpy(),
            detcov.index_select(0, inv).cpu().numpy(),
            residual.index_select(0, inv).cpu().numpy())


def mean_drop_changepoint(x):
    """Notebook cell 15, verbatim (k in [5, 30))."""
    x = np.asarray(x, float)
    n = len(x)
    sigma = np.std(x, ddof=1)
    cumsum = np.cumsum(x)
    T = np.zeros(n - 1)
    for k in range(5, 30):
        mean1 = cumsum[k - 1] / k
        mean2 = (cumsum[-1] - cumsum[k - 1]) / (n - k)
        T[k - 1] = np.sqrt(k * (n - k) / n) * (mean1 - mean2) / sigma
    tau = np.argmax(T) + 5
    return tau, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--m', type=int, default=100)
    ap.add_argument('--mcond', type=int, default=200)
    ap.add_argument('--nper', type=int, default=120)
    ap.add_argument('--probe', action='store_true',
                    help='2 training steps + 1 cycle of stats, timing only')
    ap.add_argument('--retrain', action='store_true')
    ap.add_argument('--fix-inputs', action='store_true', dest='fix_inputs',
                    help='inputs = [alt,Mach,TRA,T2] instead of the notebook '
                         'quirk [Mach,TRA,T2,unit-id]')
    args = ap.parse_args()

    tag = f'u{args.unit}' + ('' if args.seed == 0 else f'_s{args.seed}') \
        + ('_fix' if args.fix_inputs else '')
    res = os.path.join(HERE, 'colleague_repro_probe.txt' if args.probe
                       else f'colleague_repro_{tag}_results.txt')
    open(res, 'w').close()
    log(f'COLLEAGUE DEMO REPRO  DS02-006  unit={args.unit}  seed={args.seed}  '
        f'm={args.m}/{args.mcond}  steps={args.steps} lr={args.lr}  probe={args.probe}', res)

    t0 = time.time()
    W, Xs, A, W_var, Xs_var, A_var = load_dev()
    log(f'[data] dev rows={len(A)}  W_var={W_var}  outputs='
        f'{[Xs_var[i] for i in OUTPUT_IDX]}  ({time.time()-t0:.0f}s)', res)

    train_X_array, train_Y_array = build_training(W, Xs, A, args.fix_inputs)
    log(f'[data] train {train_X_array.shape} from cycle<5 pool, inputs='
        + ('[alt,Mach,TRA,T2] (FIXED)' if args.fix_inputs
           else '[Mach,TRA,T2,unit-id] (notebook iloc quirk)'), res)

    if args.probe:
        # --- timing probe: structure + 2 steps + 1 cycle ---
        xs = StandardScaler().fit(train_X_array); ys = StandardScaler().fit(train_Y_array)
        tX = torch.tensor(xs.transform(train_X_array), dtype=DT)
        tY = torch.tensor(ys.transform(train_Y_array), dtype=DT)
        model = GPyTorchMOGP(4, Q, rank=1, kernel='matern32').to(DEV, DT)
        lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Q).to(DEV, DT)
        t1 = time.time()
        r = train_vecchia_mogp(model, lik, tX, tY, m=args.m, num_steps=2, lr=args.lr,
                               group=True, verbose=True)
        dt2 = time.time() - t1
        gs = [len(g) for g in r.structure.groups]
        us = [len(set(int(v) for gg in [r.structure.neighbours[int(i)] for i in g] for v in gg))
              for g in r.structure.groups[:200]]
        log(f'[probe] structure+2 steps = {dt2:.0f}s  -> est per step '
            f'{(dt2)/2:.1f}s, {args.steps} steps ~ {dt2/2*args.steps/60:.0f} min', res)
        log(f'[probe] n_groups={len(gs)}  group size max={max(gs)}  '
            f'union size (first 200) max={max(us)}', res)
        model.eval(); lik.eval()
        unit = A[:, 0]; cycle = A[:, 1]
        rng = np.random.default_rng(args.seed)
        sel = np.where((unit == args.unit) & (cycle == 1))[0]
        sam = rng.choice(sel, size=args.nper, replace=False)
        Xfeat = make_features(W, A, args.fix_inputs)
        teX = torch.tensor(xs.transform(Xfeat[sam]), dtype=DT)
        teY = torch.tensor(ys.transform(Xs[sam][:, OUTPUT_IDX]), dtype=DT)
        t1 = time.time()
        with torch.no_grad():
            cycle_stats(model, lik, tX, tY, r.structure, teX, teY, m=args.mcond)
        max_cycle = int(cycle[unit == args.unit].max())
        log(f'[probe] 1 cycle ({args.nper} pts, mcond={args.mcond}) = {time.time()-t1:.0f}s '
            f'-> {max_cycle} cycles ~ {(time.time()-t1)*max_cycle/60:.0f} min', res)
        return

    model, lik, tX, tY, structure, xs, ys = fit_or_load(
        train_X_array, train_Y_array, args.steps, args.lr, args.m, res,
        force=args.retrain,
        ckpt_name='colleague_repro_model' + ('_fix' if args.fix_inputs else '') + '.pt')

    # ---- notebook cell 12: per-cycle stats for one unit ----
    unit = A[:, 0]; cycle = A[:, 1]; hs = A[:, 3] if len(A_var) > 3 else None
    Xfeat = make_features(W, A, args.fix_inputs)
    urows = np.where(unit == args.unit)[0]
    max_cycle = int(cycle[urows].max())
    rng = np.random.default_rng(args.seed)                       # d2

    LL, quad_l, det_l, res_l = [], [], [], []
    t0 = time.time()
    for cc in range(1, max_cycle + 1):
        sel = urows[cycle[urows] == cc]
        sam = rng.choice(sel, size=min(args.nper, len(sel)), replace=False)
        teX = torch.tensor(xs.transform(Xfeat[sam]), dtype=DT)
        teY = torch.tensor(ys.transform(Xs[sam][:, OUTPUT_IDX]), dtype=DT)
        pred, quadratic, detcov, residual = cycle_stats(
            model, lik, tX, tY, structure, teX, teY, m=args.mcond)
        quad_l.append(quadratic); det_l.append(detcov); res_l.append(residual)
        LL.append(detcov + quadratic - 0.5 * np.log(2 * np.pi))  # cell 12 verbatim
        if cc % 10 == 0 or cc == max_cycle:
            log(f'  cycle {cc}/{max_cycle}  ({time.time()-t0:.0f}s)', res)

    # ---- notebook cell 13: detcov filter ----
    LL_mean, filt_LL_mean, res_mean, filt_res_mean, kept = [], [], [], [], []
    for i in range(len(LL)):
        sel = np.where(det_l[i] > DETCOV_THR)[0]
        kept.append(len(sel))
        LL_mean.append(LL[i].mean())
        filt_LL_mean.append(LL[i][sel].mean() if len(sel) else np.nan)
        res_mean.append(res_l[i].mean(axis=0))
        filt_res_mean.append(res_l[i][sel].mean(axis=0) if len(sel) else
                             np.full(Q, np.nan))
    res_mean = np.stack(res_mean); filt_res_mean = np.stack(filt_res_mean)
    all_det = np.concatenate(det_l)
    log(f'[filter] detcov>{DETCOV_THR}: kept {np.sum(kept)}/{len(all_det)} pts '
        f'({100*np.sum(kept)/len(all_det):.1f}%)  detcov pct '
        f'[5,25,50,75,95]={np.percentile(all_det,[5,25,50,75,95]).round(2)}', res)
    log(f'[filter] kept per cycle: min={min(kept)} max={max(kept)} / {args.nper}', res)

    # ---- true onset (correct) + notebook cell 14 (faithful, buggy) ----
    hs_u = A[urows, 3]
    pos0 = np.where(hs_u == 0)[0]
    onset_correct = int(cycle[urows[pos0[0]]]) if len(pos0) else None
    onset_faithful = int(A[pos0[0], 1]) if len(pos0) else None   # d4: full-frame iloc
    log(f'[onset] correct hs==0 first cycle = {onset_correct}   '
        f'notebook cell-14 value (full-frame iloc bug) = {onset_faithful}', res)

    # ---- notebook cells 16-17: changepoint ----
    tau_raw, _ = mean_drop_changepoint(LL_mean)
    tau_fil, _ = mean_drop_changepoint(filt_LL_mean)
    log(f'[changepoint] raw LL: tau+1={tau_raw+1}   filtered LL: tau+1={tau_fil+1}   '
        f'(true onset={onset_correct})', res)

    # ---- GP prediction quality: per-cycle residual RMS, before/after filter ----
    sens = [Xs_var[i] for i in OUTPUT_IDX]
    rms_all = [float(np.sqrt((r ** 2).mean())) for r in res_l]
    rms_fil = [float(np.sqrt((r[d > DETCOV_THR] ** 2).mean())) if (d > DETCOV_THR).any()
               else np.nan for r, d in zip(res_l, det_l)]
    healthy = [c - 1 for c in range(1, max_cycle + 1)
               if onset_correct is None or c < onset_correct]
    log(f'[gp-quality] healthy-cycle residual RMS (scaled): '
        f'raw={np.nanmean([rms_all[i] for i in healthy]):.4f}  '
        f'filtered={np.nanmean([rms_fil[i] for i in healthy]):.4f}', res)
    log(f'[gp-quality] all-cycle    residual RMS (scaled): '
        f'raw={np.nanmean(rms_all):.4f}  filtered={np.nanmean(rms_fil):.4f}', res)

    np.savez_compressed(
        os.path.join(HERE, f'colleague_repro_{tag}.npz'),
        LL_mean=np.array(LL_mean), filt_LL_mean=np.array(filt_LL_mean),
        res_mean=res_mean, filt_res_mean=filt_res_mean,
        detcov_flat=np.concatenate(det_l), detcov_len=np.array([len(d) for d in det_l]),
        rms_all=np.array(rms_all), rms_fil=np.array(rms_fil),
        kept=np.array(kept), onset=np.array([onset_correct or -1]),
        tau_raw=np.array([tau_raw + 1]), tau_fil=np.array([tau_fil + 1]),
        sens=np.array(sens))

    # ---- figures (notebook cells 18-19) ----
    x = np.arange(1, len(LL_mean) + 1)
    fig = plt.figure(figsize=(11, 4))
    ax = plt.subplot(1, 2, 1)
    ax.plot(x, LL_mean, 'o', ms=4, color='gray', label='raw')
    ax.plot(x, filt_LL_mean, 'o', ms=4, color='darkblue', label=f'detcov>{DETCOV_THR}')
    ax.axvline(onset_correct, color='red', lw=2, label=f'true onset={onset_correct}')
    ax.axvline(tau_fil + 1, color='green', ls='--', label=f'changepoint={tau_fil+1}')
    ax.set_xlabel('cycle'); ax.set_ylabel('Likelihood (notebook LL)'); ax.legend(fontsize=8)
    ax.set_title(f'DS02 unit {args.unit}: per-cycle LL')
    ax = plt.subplot(1, 2, 2)
    ax.plot(x, rms_all, 'o-', ms=3, color='gray', label='raw')
    ax.plot(x, rms_fil, 'o-', ms=3, color='darkblue', label='filtered')
    ax.axvline(onset_correct, color='red', lw=2)
    ax.set_xlabel('cycle'); ax.set_ylabel('residual RMS (scaled)'); ax.legend(fontsize=8)
    ax.set_title('GP prediction error per cycle')
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, f'fig_colleague_repro_{tag}_ll.png'), dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 6))
    for j in range(Q):
        ax = plt.subplot(2, 3, j + 1)
        ax.plot(x, res_mean[:, j], '.', color='gray', label='raw mean resid')
        ax.plot(x, filt_res_mean[:, j], color='darkblue', label='filtered')
        ax.axvline(onset_correct, color='red', lw=1.5)
        ax.set_title(sens[j]); ax.legend(fontsize=7)
    plt.suptitle(f'DS02 unit {args.unit}: per-cycle mean residual (pred - actual, scaled)')
    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    plt.savefig(os.path.join(HERE, f'fig_colleague_repro_{tag}_resid.png'), dpi=130)
    plt.close(fig)

    log(f'DONE  wall={time.time()-t0:.0f}s', res)


if __name__ == '__main__':
    main()
