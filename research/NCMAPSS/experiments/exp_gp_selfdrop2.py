"""
exp_gp_selfdrop2.py — GP-ONLY validation of detcov training-set selection,
with the sample-count confound explicitly controlled. No RUL, no full stats.

Three GPs per seed, ALL trained on exactly 8192 points:
  base : 8192 random from the cycle<5 pool            (= production baseline)
  rand : 8192 random from the SAME 32768 candidates   (procedure control)
  filt : 8192 from top-50%-detcov survivors of those candidates (per
         unit-cycle group percentile, scored by the base GP)

If rand == base and filt is worse, the degradation is caused by the detcov
SELECTION itself, not by sample count or candidate provenance.

Evaluation on a common held-out normal set (4096 pts, disjoint):
  per-sensor residual RMSE, mean m2, and m2 / T48-RMSE broken down by
  detcov quartile of the held-out points (Q1 = sparsest region, Q4 = densest)
  -> shows WHERE the filtered GP fails.

Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_gp_selfdrop2.py
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from exp_ds03_port import fit_mogp, DT
from demo_cond2 import conditional_stats2

SENS = ['T48', 'T50', 'Wf']
SEEDS = [0, 1, 2]
GPD = 0.50
NPTS, NCAND, NHELD = 8192, 32768, 4096
KERNEL, RANK, M, STEPS, LR, MCOND = 'matern32', 1, 18, 120, 0.05, 15
RES = os.path.join(HERE, 'gp_selfdrop2_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def score(gp, Wq, Xq, chunk=8192):
    dcs, m2s, rss = [], [], []
    for a in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[a:a+chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[a:a+chunk]), dtype=DT)
        pred_s, dc, m2, _, _ = conditional_stats2(
            gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'], teX,
            m=MCOND, test_y=teY)
        pred = pred_s * gp['ys'].scale_ + gp['ys'].mean_
        dcs.append(dc); m2s.append(m2); rss.append(Xq[a:a+chunk] - pred)
    return np.concatenate(dcs), np.concatenate(m2s), np.concatenate(rss)


def main():
    open(RES, 'a').close()
    log(f'\n===== GP-only selfdrop validation  {time.strftime("%m-%d %H:%M")} '
        f'(all GPs trained on n={NPTS}) =====')
    cache = L.load_cache()
    sidx = [L.OUTPUT_NAMES.index(s) for s in SENS]
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    X = Xall[:, sidx]
    pool = np.where(A[:, 1].astype(int) < 5)[0]

    agg = {k: [] for k in ['base', 'rand', 'filt']}
    qagg = {k: [] for k in ['base', 'rand', 'filt']}
    for sd in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(sd)
        tr = rng.choice(pool, min(NPTS, len(pool)), replace=False)
        rest = np.setdiff1d(pool, tr)
        cand = rng.choice(rest, min(NCAND, len(rest)), replace=False)
        rest2 = np.setdiff1d(rest, cand)
        held = rng.choice(rest2, min(NHELD, len(rest2)), replace=False)

        torch.manual_seed(sd)
        gp_base = fit_mogp(W[tr], X[tr], KERNEL, RANK, M, STEPS, lr=LR)
        dc_c, _, _ = score(gp_base, W[cand], X[cand])

        grp = A[cand, 0].astype(int) * 100 + A[cand, 1].astype(int)
        keep = np.zeros(len(cand), bool)
        for g in np.unique(grp):
            m = grp == g
            thr = np.percentile(dc_c[m], GPD * 100)
            keep |= m & (dc_c >= thr)
        surv = cand[keep]
        rng2 = np.random.default_rng(sd * 1000 + 50)
        tr_f = rng2.choice(surv, NPTS, replace=False)
        rng3 = np.random.default_rng(sd * 1000 + 999)
        tr_r = rng3.choice(cand, NPTS, replace=False)

        torch.manual_seed(sd)
        gp_filt = fit_mogp(W[tr_f], X[tr_f], KERNEL, RANK, M, STEPS, lr=LR)
        torch.manual_seed(sd)
        gp_rand = fit_mogp(W[tr_r], X[tr_r], KERNEL, RANK, M, STEPS, lr=LR)

        dc_h, _, _ = score(gp_base, W[held], X[held])   # region labels (base GP)
        qedge = np.percentile(dc_h, [25, 50, 75])
        qlab = np.digitize(dc_h, qedge)                  # 0=sparsest .. 3=densest

        log(f'seed{sd}  (surv={len(surv)}, prep {time.time()-t0:.0f}s)')
        log(f'  {"GP":>5} | {"T48":>6} {"T50":>6} {"Wf":>7} | {"m2":>5} | '
            f'm2 by detcov quartile Q1(sparse)..Q4(dense)')
        for name, gp in [('base', gp_base), ('rand', gp_rand), ('filt', gp_filt)]:
            _, m2h, rsh = score(gp, W[held], X[held])
            rmse = np.sqrt((rsh ** 2).mean(0))
            qm2 = [m2h[qlab == q].mean() for q in range(4)]
            qt48 = [np.sqrt((rsh[qlab == q, 0] ** 2).mean()) for q in range(4)]
            agg[name].append(list(rmse) + [m2h.mean()])
            qagg[name].append(qm2 + qt48)
            log(f'  {name:>5} | {rmse[0]:>6.3f} {rmse[1]:>6.3f} {rmse[2]:>7.4f} | '
                f'{m2h.mean():>5.2f} | m2 {qm2[0]:.2f}/{qm2[1]:.2f}/{qm2[2]:.2f}/{qm2[3]:.2f}  '
                f'T48 {qt48[0]:.2f}/{qt48[1]:.2f}/{qt48[2]:.2f}/{qt48[3]:.2f}')

    log('\n===== 3-seed mean =====')
    log(f'  {"GP":>5} | {"T48":>6} {"T50":>6} {"Wf":>7} | {"m2":>5} | '
        f'T48 RMSE by quartile Q1(sparse)..Q4(dense)')
    for name in ['base', 'rand', 'filt']:
        a = np.mean(agg[name], 0); q = np.mean(qagg[name], 0)
        log(f'  {name:>5} | {a[0]:>6.3f} {a[1]:>6.3f} {a[2]:>7.4f} | {a[3]:>5.2f} | '
            f'{q[4]:.2f} / {q[5]:.2f} / {q[6]:.2f} / {q[7]:.2f}')
    log('done.')


if __name__ == '__main__':
    main()
