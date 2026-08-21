"""
exp_traindata_detcov.py — TRAINING-DATA cleaning by detcov, equal count & range.

Question: with the SAME training size (8192) and SAME range (cycle<5),
does removing the most-uncertain (lowest-detcov) candidate points before
training improve the GP's healthy-range prediction RMSE?

Design (per seed):
  1. candidate pool  = 16384 pts drawn from cycle<5 rows of all DS03 dev units
  2. stage-1 GP      = frozen-pipeline config trained on a plain 8192 subsample
                       (this IS the baseline / 기존 방식, drop=0)
  3. detcov of every candidate point via stage-1 GP (X-only, mcond=15)
  4. for each drop d: remove lowest d% detcov candidates -> sample 8192
                      from the remainder -> retrain GP (same config)
  5. evaluate ALL models on two FIXED common sets (identical points):
       eval-in  : 8192 cycle<5 pts excluded from the candidate pool
       eval-gen : 8192 healthy pts (cycle>=5, before each dev unit's onset)
     metric = per-sensor RMSE + z-combined RMSE (sigma from baseline eval resid)
GP config = frozen pipeline: 3 sensors T48/T50/Wf, RBF rank2, m=18,
120 steps, lr=0.1, StandardScaler.  No test-time filtering anywhere.
"""
import os, sys, time
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
SENS = os.environ.get('SENSORS', 'T48,T50,Wf').split(',')
SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
NPTS, NCAND, NEVAL = 8192, 16384, 8192
DROPS = [float(x) for x in os.environ.get(
    'DROPS', '0,0.02,0.05,0.10,0.15,0.20,0.30').split(',')]
SEEDS = [0, 1, 2]
KERNEL = os.environ.get('KERNEL', 'rbf')
RANK = int(os.environ.get('RANK', 2))
M, STEPS, LR, MCOND = 18, 120, 0.1, 15
TRAIN_CYC = 5
TAG = os.environ.get('TAG', '')
RES = os.path.join(HERE, f'traindata_detcov{TAG}_results.txt')
NPZ = os.path.join(HERE, f'traindata_detcov{TAG}_sweep.npz')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def fit_gp(Wtr, Ytr):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT)
    tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=len(SENS), rank=RANK, kernel=KERNEL).to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=len(SENS)).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=M, num_steps=STEPS, lr=LR,
                           group=True, verbose=False)
    model.eval(); lik.eval()
    noise = float(lik.noise.detach())
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure,
                xs=xs, ys=ys, noise=noise)


@torch.no_grad()
def predict(gp, Wq, chunk=4096):
    """Vecchia conditional mean at query points, raw units."""
    out = []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        pred_s, *_ = conditional_stats2(gp['model'], gp['lik'], gp['tX'], gp['tY'],
                                        gp['struct'], teX, m=MCOND, test_y=None)
        out.append(pred_s * gp['ys'].scale_ + gp['ys'].mean_)
    return np.concatenate(out)


@torch.no_grad()
def detcov_of(gp, Wq, chunk=4096):
    out = []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        _, dc, *_ = conditional_stats2(gp['model'], gp['lik'], gp['tX'], gp['tY'],
                                       gp['struct'], teX, m=MCOND, test_y=None)
        out.append(dc)
    return np.concatenate(out)


def true_onset_mask(A):
    """healthy mask for cycle>=TRAIN_CYC rows of dev units (before hs onset)."""
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int); hs = A[:, 3]
    mask = np.zeros(len(A), bool)
    for u in np.unique(unit):
        ur = np.where(unit == u)[0]
        ucyc = np.unique(cyc[ur])
        hs_by = np.array([hs[ur[cyc[ur] == c]].mean() for c in ucyc])
        below = np.where(hs_by < 0.5)[0]
        onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
        mask[ur] = (cyc[ur] >= TRAIN_CYC) & (cyc[ur] < onset)
    return mask


def main():
    open(RES, 'w').close()
    log(f'TRAIN-DATA DETCOV CLEANING SWEEP  drops={DROPS}  seeds={SEEDS}')
    log(f'  equal count ({NPTS}) & range (cycle<{TRAIN_CYC});  eval on fixed common sets;'
        f'  no test-time filtering')
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    cyc = A[:, 1].astype(int)
    pool = np.where(cyc < TRAIN_CYC)[0]
    healthy_gen = np.where(true_onset_mask(A))[0]
    log(f'  pool cycle<{TRAIN_CYC}: {len(pool)} rows | healthy gen rows: {len(healthy_gen)}')

    results = {}
    t00 = time.time()
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        perm = rng.permutation(len(pool))
        cand = pool[perm[:NCAND]]
        evin = pool[perm[NCAND:NCAND + NEVAL]]              # held-out, cycle<5
        evgn = rng.choice(healthy_gen, NEVAL, replace=False)  # healthy, cycle>=5
        Yin, Ygn = X[evin], X[evgn]

        # stage 1: baseline GP on plain 8192 subsample of candidates
        base_tr = cand[rng.choice(NCAND, NPTS, replace=False)]
        t0 = time.time()
        gp0 = fit_gp(W[base_tr], X[base_tr])
        dc_cand = detcov_of(gp0, W[cand])
        log(f'seed{sd}: stage-1 GP + candidate detcov done ({time.time()-t0:.0f}s)  '
            f'noise={gp0["noise"]:.4f}  detcov pct[5,50,95]='
            f'{np.percentile(dc_cand, [5, 50, 95]).round(3)}')

        for dr in DROPS:
            t0 = time.time()
            if dr == 0.0:
                gp = gp0                                     # 기존 방식 그대로
            else:
                thr = np.percentile(dc_cand, dr * 100)
                keep = cand[dc_cand >= thr]
                rng_d = np.random.default_rng(sd * 100 + int(dr * 100))
                tr = keep[rng_d.choice(len(keep), NPTS, replace=False)]
                gp = fit_gp(W[tr], X[tr])
            r_in = Yin - predict(gp, W[evin])
            r_gn = Ygn - predict(gp, W[evgn])
            if dr == 0.0:
                zin = r_in.std(0); zgn = r_gn.std(0)         # fixed sigma from baseline
                results[f's{sd}_zin'] = zin; results[f's{sd}_zgn'] = zgn
            zin = results[f's{sd}_zin']; zgn = results[f's{sd}_zgn']
            row = dict(
                rmse_in=np.sqrt((r_in ** 2).mean(0)), rmse_gn=np.sqrt((r_gn ** 2).mean(0)),
                z_in=float(np.sqrt(((r_in / zin) ** 2).mean())),
                z_gn=float(np.sqrt(((r_gn / zgn) ** 2).mean())),
                noise=gp['noise'])
            results[f's{sd}_d{int(dr*100)}'] = row
            log(f'  seed{sd} drop={dr:>4.0%}: held-in zRMSE={row["z_in"]:.4f} '
                f'(T48 {row["rmse_in"][0]:.3f})  healthy-gen zRMSE={row["z_gn"]:.4f} '
                f'(T48 {row["rmse_gn"][0]:.3f})  noise={row["noise"]:.4f} '
                f'({time.time()-t0:.0f}s)')

    # ---- summary ----
    log('')
    log(f'{"drop":>6} | {"held-in zRMSE (cyc<5)":>22} | {"healthy-gen zRMSE":>18} | '
        f'{"T48 in":>7} | {"T48 gen":>7} | {"noise":>6}')
    log('-' * 84)
    npz = {}
    for dr in DROPS:
        zi = [results[f's{sd}_d{int(dr*100)}']['z_in'] for sd in SEEDS]
        zg = [results[f's{sd}_d{int(dr*100)}']['z_gn'] for sd in SEEDS]
        t48i = np.mean([results[f's{sd}_d{int(dr*100)}']['rmse_in'][0] for sd in SEEDS])
        t48g = np.mean([results[f's{sd}_d{int(dr*100)}']['rmse_gn'][0] for sd in SEEDS])
        nz = np.mean([results[f's{sd}_d{int(dr*100)}']['noise'] for sd in SEEDS])
        log(f'{dr:>6.0%} | {np.mean(zi):>10.4f} ± {np.std(zi):<8.4f} | '
            f'{np.mean(zg):>8.4f} ± {np.std(zg):<6.4f} | {t48i:>7.3f} | {t48g:>7.3f} | {nz:>6.4f}')
        npz[f'd{int(dr*100)}_zin'] = np.array(zi)
        npz[f'd{int(dr*100)}_zgn'] = np.array(zg)
    np.savez(NPZ, drops=np.array(DROPS), **npz)
    log(f'\nwall total = {(time.time()-t00)/60:.1f} min   -> {NPZ}')


if __name__ == '__main__':
    main()
