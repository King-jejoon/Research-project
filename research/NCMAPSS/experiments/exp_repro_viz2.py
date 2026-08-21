"""
exp_repro_viz2.py — readable single-sensor view of the detcov inspection.

Layout (shared x = time within cycle):
  (a) sensor trace: full actual data line + altitude background, kept points
      green, discarded red, zoom inset on the discard region
  (b) prediction error (actual - GP pred) at the sampled points, own scale
  (c) detcov with threshold line
Reuses the sampled-point computation (same seed) — no full-cycle GP pass.
"""
import os, sys, argparse
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))
import gpytorch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __import__('exp_paths').DEMO)
from sklearn.preprocessing import StandardScaler
from gpytorch_mogp_vecchia import GPyTorchMOGP
from exp_colleague_repro import (load_dev, make_features, build_training,
                                 cycle_stats, OUTPUT_IDX, DETCOV_THR, Q, DT, DEV)

ap = argparse.ArgumentParser()
ap.add_argument('--unit', type=int, default=10)
ap.add_argument('--cycle', type=int, default=12)
ap.add_argument('--sensor', default='T48')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--nper', type=int, default=120)
ap.add_argument('--mcond', type=int, default=200)
ap.add_argument('--faithful', action='store_true')
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
j = sens.index(args.sensor)
unit_lbl = {'T30': '°R', 'T48': '°R', 'T50': '°R', 'Nc': 'rpm', 'Wf': 'pps'}[args.sensor]

tmin = (sam - rows[0]) / 60.0
tfull = np.arange(len(rows)) / 60.0
full = Xs[rows][:, OUTPUT_IDX[j]]
alt = W[rows, 0]
err = actual[:, j] - pred[:, j]

fig, (axA, axB, axC) = plt.subplots(
    3, 1, figsize=(14, 10), sharex=True,
    gridspec_kw={'height_ratios': [2.6, 1, 1], 'hspace': 0.08})

# ---- (a) sensor trace ----
axAlt = axA.twinx()
axAlt.fill_between(tfull, alt / 1000.0, color='skyblue', alpha=0.18, zorder=0)
axAlt.set_ylabel('altitude (kft)', color='steelblue')
axAlt.tick_params(axis='y', labelcolor='steelblue')
axA.plot(tfull, full, '-', lw=0.8, color='0.45', zorder=2, label='actual data (full cycle)')
axA.plot(tmin[keep], actual[keep, j], 'o', ms=6, color='tab:green', zorder=4,
         mec='white', mew=0.5, label='kept points')
axA.plot(tmin[drop], actual[drop, j], 'o', ms=9, color='red', zorder=5,
         mec='darkred', mew=0.8, label=f'discarded (detcov ≤ {DETCOV_THR})')
axA.set_ylabel(f'{args.sensor} ({unit_lbl})')
axA.set_title(f'DS02 unit {args.unit}, cycle {args.cycle} — {args.sensor} — '
              f'{int(drop.sum())}/{args.nper} sampled points discarded', fontsize=13)
axA.legend(loc='upper right', fontsize=9)
axA.set_zorder(axAlt.get_zorder() + 1)
axA.patch.set_visible(False)

# zoom inset around the discard region
if drop.any():
    lo, hi = tmin[drop].min() - 4, tmin[drop].max() + 4
    axin = inset_axes(axA, width='38%', height='45%', loc='lower left',
                      bbox_to_anchor=(0.04, 0.06, 1, 1), bbox_transform=axA.transAxes)
    mfull = (tfull >= lo) & (tfull <= hi)
    axin.plot(tfull[mfull], full[mfull], '-', lw=0.9, color='0.45')
    mk = keep & (tmin >= lo) & (tmin <= hi)
    md = drop & (tmin >= lo) & (tmin <= hi)
    axin.plot(tmin[mk], actual[mk, j], 'o', ms=6, color='tab:green', mec='white', mew=0.5)
    axin.plot(tmin[mk], pred[mk, j], '_', ms=8, mew=1.6, color='tab:blue')
    axin.plot(tmin[md], actual[md, j], 'o', ms=9, color='red', mec='darkred', mew=0.8)
    axin.plot(tmin[md], pred[md, j], '_', ms=8, mew=1.6, color='tab:blue',
              label='GP prediction')
    axin.set_xlim(lo, hi)
    ypad = 0.08 * (full[mfull].max() - full[mfull].min())
    axin.set_ylim(full[mfull].min() - ypad, full[mfull].max() + ypad)
    axin.tick_params(labelsize=7)
    axin.set_title('zoom: discard region (─ = GP prediction)', fontsize=8)
    mark_inset(axA, axin, loc1=1, loc2=3, fc='none', ec='0.6', lw=0.7)

# ---- (b) prediction error ----
axB.axhline(0, color='0.6', lw=0.8)
axB.vlines(tmin[keep], 0, err[keep], color='tab:green', lw=1, alpha=0.7)
axB.vlines(tmin[drop], 0, err[drop], color='red', lw=1.4, alpha=0.85)
axB.plot(tmin[keep], err[keep], 'o', ms=4, color='tab:green')
axB.plot(tmin[drop], err[drop], 'o', ms=6, color='red', mec='darkred', mew=0.7)
axB.set_ylabel(f'actual − pred ({unit_lbl})')

# ---- (c) detcov ----
axC.plot(tmin[keep], detcov[keep], 'o', ms=4, color='tab:green', label='kept')
axC.plot(tmin[drop], detcov[drop], 'o', ms=6, color='red', mec='darkred', mew=0.7,
         label='discarded')
axC.axhline(DETCOV_THR, color='red', ls='--', lw=1.2, label=f'threshold {DETCOV_THR}')
axC.set_ylabel('detcov')
axC.set_xlabel('time within cycle (min)')
axC.legend(loc='lower left', fontsize=9)

out = os.path.join(HERE, f'fig_viz2_u{args.unit}_c{args.cycle}_{args.sensor}'
                   + ('_fix' if fix else '') + '.png')
plt.savefig(out, dpi=130, bbox_inches='tight')
print('saved', out)
print(f'|err| kept mean={np.abs(err[keep]).mean():.2f}  '
      f'discarded mean={np.abs(err[drop]).mean():.2f} {unit_lbl}')
