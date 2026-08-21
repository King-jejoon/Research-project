"""
exp_repro_viz.py — DS02 unit10 한 사이클의 실측/예측/버려진 점 시각화.

저장된 colleague_repro 체크포인트(기본: 수정판 alt 포함)를 재사용해
한 사이클 120점의 GP 조건부 예측을 다시 계산하고, 원 단위로 되돌려
센서 5개 + detcov 패널에 그린다.  버려진 점 = detcov <= 3.60.
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
ap.add_argument('--faithful', action='store_true',
                help='use the faithful (alt-dropped) checkpoint instead')
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
pred = pred_s * ys.scale_ + ys.mean_                      # back to raw units
actual = Xs[sam][:, OUTPUT_IDX]
drop = detcov <= DETCOV_THR
keep = ~drop
tmin = (sam - rows[0]) / 60.0                             # minutes within cycle
full = Xs[rows][:, OUTPUT_IDX]                            # the real data, ALL points
tfull = np.arange(len(rows)) / 60.0

sens = [str(Xs_var[i]) for i in OUTPUT_IDX]
unit_lbl = ['°R', '°R', '°R', 'rpm', 'pps']
fig = plt.figure(figsize=(15, 8))
for j in range(Q):
    ax = plt.subplot(2, 3, j + 1)
    ax.plot(tfull, full[:, j], '-', lw=0.7, color='0.55', label='actual data (full cycle)')
    ax.plot(tmin[keep], actual[keep, j], 'o', ms=5, color='tab:green',
            label='used points (actual)')
    ax.plot(tmin, pred[:, j], 'o', ms=3.5, color='tab:blue', label='GP prediction')
    ax.plot(tmin[drop], actual[drop, j], 'o', ms=5, color='red',
            label=f'discarded (detcov ≤ {DETCOV_THR})')
    ax.set_title(sens[j]); ax.set_ylabel(unit_lbl[j])
    if j == 0:
        ax.legend(fontsize=8)
ax = plt.subplot(2, 3, 6)
ax.plot(tmin[keep], detcov[keep], 'o', ms=5, color='tab:green', label='kept')
ax.plot(tmin[drop], detcov[drop], 'o', ms=5, color='red', label='discarded')
ax.axhline(DETCOV_THR, color='red', ls='--', lw=1, label=f'threshold {DETCOV_THR}')
ax.set_title('detcov (lower = more uncertain)')
ax.legend(fontsize=8)
for axx in fig.axes:
    axx.set_xlabel('time within cycle (min)')
mode = 'fixed inputs (alt included)' if fix else 'faithful (alt dropped)'
plt.suptitle(f'DS02 unit {args.unit}, cycle {args.cycle} ({mode}): '
             f'{int(drop.sum())}/{args.nper} sampled points discarded', fontsize=13)
plt.tight_layout()
out = os.path.join(HERE, f'fig_viz_u{args.unit}_c{args.cycle}'
                   + ('_fix' if fix else '') + '.png')
plt.savefig(out, dpi=130)
print('saved', out)
print('detcov range', detcov.min().round(3), detcov.max().round(3),
      '| dropped', int(drop.sum()), '/', len(drop))
rr = actual - pred
print('cycle RMSE raw-units per sensor:',
      dict(zip(sens, np.sqrt((rr ** 2).mean(0)).round(3))))
kk = ~drop
print('filtered RMSE               :',
      dict(zip(sens, np.sqrt((rr[kk] ** 2).mean(0)).round(3))))
