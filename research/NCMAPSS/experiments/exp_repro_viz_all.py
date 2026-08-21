"""
exp_repro_viz_all.py — all 5 sensors stacked, shared time axis.
Each panel: full actual data line + altitude background, kept sampled points
green, discarded red.  Bottom panel: detcov with threshold.
Discard decision is per time point (joint over all 5 outputs), so red points
line up vertically across sensors.
"""
import os, sys, argparse
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
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--nper', type=int, default=120)
ap.add_argument('--mcond', type=int, default=200)
ap.add_argument('--faithful', action='store_true')
ap.add_argument('--tmin', type=float, default=None, help='zoom window start (min)')
ap.add_argument('--tmax', type=float, default=None, help='zoom window end (min)')
args = ap.parse_args()
fix = not args.faithful

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
rng = np.random.default_rng(args.seed)
sam = np.sort(rng.choice(rows, size=min(args.nper, len(rows)), replace=False))
Xfeat = make_features(W, A, fix)
teX = torch.tensor(xs.transform(Xfeat[sam]), dtype=DT)
teY = torch.tensor(ys.transform(Xs[sam][:, OUTPUT_IDX]), dtype=DT)
pred_s, quad, detcov, resid = cycle_stats(model, lik, tX, tY, structure, teX, teY,
                                          m=args.mcond)
pred = pred_s * ys.scale_ + ys.mean_
actual = Xs[sam][:, OUTPUT_IDX]
drop = detcov <= DETCOV_THR
keep = ~drop

sens = [str(Xs_var[i]) for i in OUTPUT_IDX]
unit_lbl = ['°R', '°R', '°R', 'rpm', 'pps']
tmin = (sam - rows[0]) / 60.0
tfull = np.arange(len(rows)) / 60.0
alt = W[rows, 0] / 1000.0

fig, axes = plt.subplots(Q + 1, 1, figsize=(14, 16), sharex=True,
                         gridspec_kw={'height_ratios': [1] * Q + [0.8],
                                      'hspace': 0.10})
for j in range(Q):
    ax = axes[j]
    ax2 = ax.twinx()
    ax2.fill_between(tfull, alt, color='skyblue', alpha=0.15, zorder=0)
    ax2.set_yticks([])
    ax.plot(tfull, Xs[rows][:, OUTPUT_IDX[j]], '-', lw=0.8, color='0.45', zorder=2,
            label='actual data (full cycle)')
    ax.plot(tmin[keep], actual[keep, j], 'o', ms=5, color='tab:green', zorder=4,
            mec='white', mew=0.4, label='kept points')
    ax.plot(tmin[drop], actual[drop, j], 'o', ms=8, color='red', zorder=5,
            mec='darkred', mew=0.7, label=f'discarded (detcov ≤ {DETCOV_THR})')
    for t in tmin[drop]:
        ax.axvline(t, color='red', alpha=0.08, lw=4, zorder=1)
    ax.set_ylabel(f'{sens[j]} ({unit_lbl[j]})')
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    if j == 0:
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(f'DS02 unit {args.unit}, cycle {args.cycle} — all 5 sensors — '
                     f'{int(drop.sum())}/{args.nper} points discarded '
                     f'(shaded background = altitude)', fontsize=13)

ax = axes[Q]
ax.plot(tmin[keep], detcov[keep], 'o', ms=4, color='tab:green', label='kept')
ax.plot(tmin[drop], detcov[drop], 'o', ms=7, color='red', mec='darkred', mew=0.7,
        label='discarded')
for t in tmin[drop]:
    ax.axvline(t, color='red', alpha=0.08, lw=4, zorder=1)
ax.axhline(DETCOV_THR, color='red', ls='--', lw=1.2, label=f'threshold {DETCOV_THR}')
ax.set_ylabel('detcov')
ax.set_xlabel('time within cycle (min)')
ax.legend(loc='lower left', fontsize=9)

zoom = ''
if args.tmin is not None or args.tmax is not None:
    lo = args.tmin if args.tmin is not None else tfull[0]
    hi = args.tmax if args.tmax is not None else tfull[-1]
    m = (tfull >= lo) & (tfull <= hi)
    for j in range(Q):
        yv = Xs[rows][:, OUTPUT_IDX[j]][m]
        pad = 0.06 * (yv.max() - yv.min())
        axes[j].set_ylim(yv.min() - pad, yv.max() + pad)
    axes[0].set_xlim(lo, hi)
    zoom = f'_z{int(lo)}-{int(hi)}'
    axes[0].set_title(f'DS02 unit {args.unit}, cycle {args.cycle} — zoom '
                      f'{lo:.0f}-{hi:.0f} min', fontsize=13)

out = os.path.join(HERE, f'fig_vizall_u{args.unit}_c{args.cycle}'
                   + ('_fix' if fix else '') + zoom + '.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print('saved', out)
