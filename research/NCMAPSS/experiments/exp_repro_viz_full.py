"""
exp_repro_viz_full.py — DS02 unit10 one full cycle: ALL 1Hz points drawn as
lines (actual + GP prediction), discarded points (detcov<=3.60) as red dots.

Computes conditional stats for every point of the cycle (~8-9k pts, a few
minutes).  Stats are cached to npz so re-plotting is instant (--replot).
All figure text in English.
"""
import os, sys, time, argparse
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __import__('exp_paths').DEMO)
from sklearn.preprocessing import StandardScaler
from gpytorch_mogp_vecchia import GPyTorchMOGP
from exp_colleague_repro import (load_dev, make_features, build_training,
                                 cycle_stats, OUTPUT_IDX, DETCOV_THR, Q, DT, DEV)

ap = argparse.ArgumentParser()
ap.add_argument('--unit', type=int, default=10)
ap.add_argument('--cycle', type=int, default=12)
ap.add_argument('--mcond', type=int, default=200)
ap.add_argument('--stride', type=int, default=1)
ap.add_argument('--faithful', action='store_true')
ap.add_argument('--replot', action='store_true', help='reuse cached stats npz')
args = ap.parse_args()
fix = not args.faithful

tag = f'u{args.unit}_c{args.cycle}' + ('_fix' if fix else '')
npz_path = os.path.join(HERE, f'vizfull_{tag}.npz')
SENS_LBL = None

t0 = time.time()
if args.replot and os.path.exists(npz_path):
    z = np.load(npz_path)
    actual, pred, detcov = z['actual'], z['pred'], z['detcov']
    sens = [str(s) for s in z['sens']]
    stride = int(z['stride'][0])
else:
    W, Xs, A, W_var, Xs_var, A_var = load_dev()
    train_X_array, train_Y_array = build_training(W, Xs, A, fix)
    xs = StandardScaler().fit(train_X_array)
    ys = StandardScaler().fit(train_Y_array)
    tX = torch.tensor(xs.transform(train_X_array), dtype=DT)
    tY = torch.tensor(ys.transform(train_Y_array), dtype=DT)

    ckpt = os.path.join(HERE, 'colleague_repro_model' + ('_fix' if fix else '') + '.pt')
    blob = torch.load(ckpt, weights_only=False)
    model = GPyTorchMOGP(4, Q, rank=1, kernel='matern32').to(DEV, DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Q).to(DEV, DT)
    model.load_state_dict(blob['model']); lik.load_state_dict(blob['lik'])
    model.eval(); lik.eval()
    structure = blob['structure']

    unit, cycle = A[:, 0], A[:, 1]
    rows = np.where((unit == args.unit) & (cycle == args.cycle))[0]
    rows = rows[::args.stride]
    Xfeat = make_features(W, A, fix)
    teX = torch.tensor(xs.transform(Xfeat[rows]), dtype=DT)
    teY = torch.tensor(ys.transform(Xs[rows][:, OUTPUT_IDX]), dtype=DT)
    print(f'cycle {args.cycle}: {len(rows)} pts, computing conditional stats ...', flush=True)
    pred_s, quad, detcov, resid = cycle_stats(model, lik, tX, tY, structure, teX, teY,
                                              m=args.mcond)
    print(f'stats done ({time.time()-t0:.0f}s)', flush=True)
    pred = pred_s * ys.scale_ + ys.mean_
    actual = Xs[rows][:, OUTPUT_IDX]
    sens = [str(Xs_var[i]) for i in OUTPUT_IDX]
    stride = args.stride
    np.savez_compressed(npz_path, actual=actual, pred=pred, detcov=detcov,
                        sens=np.array(sens), stride=np.array([stride]))

drop = detcov <= DETCOV_THR
tmin = np.arange(len(actual)) * stride / 60.0
unit_lbl = ['°R', '°R', '°R', 'rpm', 'pps']

fig = plt.figure(figsize=(15, 8))
for j in range(Q):
    ax = plt.subplot(2, 3, j + 1)
    ax.plot(tmin, actual[:, j], '-', lw=0.9, color='0.3', label='actual (all points)')
    ax.plot(tmin, pred[:, j], '-', lw=0.8, color='tab:blue', alpha=0.75,
            label='GP prediction')
    ax.plot(tmin[drop], actual[drop, j], '.', ms=5, color='red',
            label=f'discarded (detcov ≤ {DETCOV_THR})')
    ax.set_title(sens[j]); ax.set_ylabel(unit_lbl[j])
    if j == 0:
        ax.legend(fontsize=8, loc='best')
ax = plt.subplot(2, 3, 6)
ax.plot(tmin, detcov, '-', lw=0.8, color='tab:green', label='detcov')
ax.plot(tmin[drop], detcov[drop], '.', ms=5, color='red')
ax.axhline(DETCOV_THR, color='red', ls='--', lw=1, label=f'threshold {DETCOV_THR}')
ax.set_title('detcov (lower = more uncertain)')
ax.legend(fontsize=8)
for axx in fig.axes:
    axx.set_xlabel('time within cycle (min)')
mode = 'fixed inputs (alt included)' if fix else 'faithful (alt dropped)'
plt.suptitle(f'DS02 unit {args.unit}, cycle {args.cycle} — {mode} — '
             f'{int(drop.sum())}/{len(drop)} points discarded '
             f'({100*drop.mean():.1f}%)', fontsize=13)
plt.tight_layout()
out = os.path.join(HERE, f'fig_vizfull_{tag}.png')
plt.savefig(out, dpi=130)
print('saved', out)
rr = actual - pred
print('full-cycle RMSE :', dict(zip(sens, np.sqrt((rr ** 2).mean(0)).round(3))))
print(f'dropped {int(drop.sum())}/{len(drop)}  wall={time.time()-t0:.0f}s')
