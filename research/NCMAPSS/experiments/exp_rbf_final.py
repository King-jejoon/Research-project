"""
exp_rbf_final.py — full re-optimization with kernel = RBF rank 2 (GP-metric
selection), following the corrected protocol:

  [A] lr for rbf-r2 chosen by GP-level held-out normal fit (DS03 dev cycle<5)
  [B] DS03 stats caches (NPER=200) at the winning lr
  [C] downstream re-descent on DS03 ONLY (no filter, exp Stage 5):
      lambda_mono x lambda_conv grid -> flat grid   (shape-pass, min NASA score)
  [D] DS01/DS02 stats caches with the FROZEN configuration
  [E] pure transfer evaluation: official 20/40/60/80 + 10% truncation grid

Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_rbf_final.py
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
import exp_ds03_port as EP
from exp_ds03_port import fit_mogp, nasa_score, FRACS
from exp_gp_selfdrop import build_resid, score_points, detect_acc
from neural_fusion_tail import train_model_tail
from stage5_exp import rul_eval_exp
from exp_resweep_mono import shape_of

SEEDS = [0, 1, 2]
KERNEL, RANK = 'rbf', 2
RES = os.path.join(HERE, 'rbf_final_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def hi_eval(RD, lm, lc, fw, fm, fracs=FRACS):
    rmses, scores, mets = [], [], []
    for sd in SEEDS:
        dl, tl = RD[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                  lambda1=lm, lambda2=lc, init_threshold=0.2,
                                  flat_w=fw, flat_m=fm, alpha=0.001)
        dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
        tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
        P, T, FR, UN, beta = rul_eval_exp(dHI, tHI, fracs)
        rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
        scores.append(nasa_score(P, T))
        for u in tHI: mets.append(shape_of(tHI[u]))
    st, s9, s10, en = np.nanmean(np.array(mets), 0)
    ok = (s10 >= s9) and (st <= 0.30) and (en >= 0.93)
    return np.mean(rmses), np.std(rmses), np.mean(scores), np.std(scores), ok


def main():
    open(RES, 'w').close()
    t0 = time.time()
    cache03 = L.load_cache()
    sidx = [L.OUTPUT_NAMES.index(s) for s in ['T48', 'T50', 'Wf']]
    W, Xall, A = cache03['W_dev'], cache03['X_s_dev'], cache03['A_dev']
    X = Xall[:, sidx]
    pool = np.where(A[:, 1].astype(int) < 5)[0]

    # ---- [A] lr by GP-level held-out fit ----
    log('[A] rbf-r2 lr selection by GP-level held-out normal RMSE (standardized, pooled)')
    best_lr = None
    for lr in [0.05, 0.1]:
        pooled = []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            tr = rng.choice(pool, 8192, replace=False)
            rest = np.setdiff1d(pool, tr)
            held = rng.choice(rest, 4096, replace=False)
            torch.manual_seed(sd)
            gp = fit_mogp(W[tr], X[tr], KERNEL, RANK, 18, 120, lr=lr)
            _, _, resid = score_points(gp, W[held], X[held])
            z = resid / gp['ys'].scale_
            pooled.append(float(np.sqrt((z ** 2).mean())))
        m, s = np.mean(pooled), np.std(pooled)
        log(f'  lr={lr}: pooled standardized heldout RMSE {m:.4f} ± {s:.4f}  ({time.time()-t0:.0f}s)')
        if best_lr is None or m < best_lr[1]: best_lr = (lr, m)
    LR = best_lr[0]
    log(f'--> lr winner (GP metric): {LR}')

    # ---- [B] DS03 stats caches ----
    EP.NPER = 200
    tag03 = f's3_rbf_r2_m18_n8192_p200_lr{LR}'
    cfg = dict(sens=['T48', 'T50', 'Wf'], sidx=sidx, kernel=KERNEL, rank=RANK, m=18,
               npts=8192, steps=120, mcond=15, tag=tag03, lr=LR)
    zs03 = {}
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        path = os.path.join(HERE, f'port_stats_{tag03}_s{sd}.npz')
        zs03[sd] = EP.compute_stats(cache03, sd, cfg, path)
        log(f'[B] DS03 seed{sd} stats ready ({time.time()-t0:.0f}s)')

    # ---- [C] downstream descent on DS03 ----
    RD = {sd: build_resid(zs03[sd], 0.0) for sd in SEEDS}
    log('[C] lambda grid (flat 300/.002)')
    best = None
    for lm in [0.5, 1, 2, 4, 6, 8]:
        for lc in [0.25, 0.5, 1, 2, 4]:
            r, rs, sc, ss, ok = hi_eval(RD, lm, lc, 300.0, 0.002)
            log(f'  {lm} {lc}: RMSE {r:.3f} ±{rs:.3f}  score {sc:.3f}  {"PASS" if ok else "fail"}')
            if ok and (best is None or sc < best[2] - 1e-9 or (abs(sc - best[2]) < 1e-9 and r < best[3])):
                best = (lm, lc, sc, r)
    log(f'--> lambda winner: {best[0]}, {best[1]}')
    log('[C] flat grid')
    bestF = None
    for fw in [300, 500, 800, 1200]:
        for fm in [0.002, 0.004, 0.006]:
            r, rs, sc, ss, ok = hi_eval(RD, best[0], best[1], float(fw), fm)
            log(f'  {fw} {fm}: RMSE {r:.3f} ±{rs:.3f}  score {sc:.3f}  {"PASS" if ok else "fail"}')
            if ok and (bestF is None or sc < bestF[2] - 1e-9 or (abs(sc - bestF[2]) < 1e-9 and r < bestF[3])):
                bestF = (fw, fm, sc, r)
    log(f'--> flat winner: {bestF[0]}, {bestF[1]}')
    LM, LC, FW, FM = best[0], best[1], float(bestF[0]), bestF[1]
    log(f'FROZEN CONFIG: rbf r2, lr={LR}, lambda=({LM},{LC}), flat=({FW:.0f},{FM}), no filter, exp S5')

    # ---- [D] DS01/DS02 caches with frozen config ----
    zs = {'DS03': zs03}
    for name, cpath in [('DS01', 'cache_ds01.npz'), ('DS02', 'cache_ds02.npz')]:
        c = dict(np.load(os.path.join(HERE, cpath), allow_pickle=False))
        tag = f'ds{name[-1]}_s3_rbf_r2_m18_n8192_p200_lr{LR}'
        zz = {}
        for sd in SEEDS:
            torch.manual_seed(sd); np.random.seed(sd)
            path = os.path.join(HERE, f'port_stats_{tag}_s{sd}.npz')
            zz[sd] = EP.compute_stats(c, sd, cfg, path)
            log(f'[D] {name} seed{sd} stats ready ({time.time()-t0:.0f}s)')
        zs[name] = zz

    # ---- [E] final evaluation ----
    FR5 = [round(f, 2) for f in np.arange(0.05, 0.96, 0.05)]
    out = {}
    for name in ['DS03', 'DS01', 'DS02']:
        rows = {f: [[], []] for f in FR5}
        rmses, scores, betas = [], [], []
        for sd in SEEDS:
            dl, tl = build_resid(zs[name][sd], 0.0)
            np.random.seed(sd)
            him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                      lambda1=LM, lambda2=LC, init_threshold=0.2,
                                      flat_w=FW, flat_m=FM, alpha=0.001)
            dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
            tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
            P, T, FR, UN, beta = rul_eval_exp(dHI, tHI, FR5)
            betas.append(beta)
            m4 = np.isin(FR, FRACS)
            rmses.append(float(np.sqrt(((P[m4] - T[m4]) ** 2).mean())))
            scores.append(nasa_score(P[m4], T[m4]))
            for f in FR5:
                m = FR == f
                rows[f][0].append(float(np.sqrt(((P[m] - T[m]) ** 2).mean())))
                rows[f][1].append(nasa_score(P[m], T[m]))
        out[name] = {f: (np.mean(rows[f][0]), np.std(rows[f][0]), np.mean(rows[f][1])) for f in FR5}
        a, dly = np.mean([detect_acc(zs[name][sd]) for sd in SEEDS], 0)
        log(f'[E] {name}: RMSE {np.mean(rmses):.2f} ±{np.std(rmses):.2f}  '
            f'score {np.mean(scores):.1f} ±{np.std(scores):.1f}  betas={betas}  '
            f'det {a:.1f}%/{dly:.1f}cyc')
    np.save(os.path.join(HERE, 'rbf_final_trunc.npy'), out, allow_pickle=True)
    log(f'DONE {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
