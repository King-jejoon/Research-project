"""
exp_resweep_mono.py — downstream re-validation under the REVISED Stage 5
(monotone basis [1, c^2]), DS03, 3 seeds. Coordinate descent in pipeline order:

  [A] lambda_mono x lambda_conv grid   (L_flat(300,.002), drop 60%)
  [B] flat_w x margin grid             (at A winner)
  [C] threshold sweep 0-95%            (at final HI)

Selection rule unchanged: shape-first (start<=0.30, s10>=s9, end>=0.93),
then lowest NASA score among passers.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_resweep_mono.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_gp_selfdrop import build_resid
from exp_ds03_port import nasa_score, FRACS
from stage5_mono import rul_eval_mono
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'resweep_mono_results.txt')
CACHE = 'port_stats_s3_matern32_r1_m18_n8192_p200_s{sd}.npz'


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def shape_of(h):
    hs = smooth1d(h); n = len(hs); x = np.linspace(0, 1, n)
    start = hs[x <= 0.10].mean() if (x <= 0.10).any() else hs[0]
    s9m = (x >= 0.8) & (x <= 0.9); s10m = (x >= 0.9)
    s9 = np.polyfit(x[s9m], hs[s9m], 1)[0]; s10 = np.polyfit(x[s10m], hs[s10m], 1)[0]
    return start, s9, s10, hs[-1]


def evaluate(RD, lm, lc, fw, fm, fracs=FRACS):
    rmses, scores, mets = [], [], []
    for sd in SEEDS:
        dl, tl = RD[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                  lambda1=lm, lambda2=lc, init_threshold=0.2,
                                  flat_w=fw, flat_m=fm, alpha=0.001)
        dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
        tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
        P, T, FR, UN = rul_eval_mono(dHI, tHI, fracs)
        rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
        scores.append(nasa_score(P, T))
        for u in tHI: mets.append(shape_of(tHI[u]))
    start, s9, s10, end = np.nanmean(np.array(mets), 0)
    ok = (s10 >= s9) and (start <= 0.30) and (end >= 0.93)
    return np.mean(rmses), np.std(rmses), np.mean(scores), s9, s10, ok


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('DOWNSTREAM RE-SWEEP under REVISED Stage 5 (basis [1, c^2]) — DS03, 3 seeds')
    z = {sd: np.load(os.path.join(HERE, CACHE.format(sd=sd))) for sd in SEEDS}

    # ---- A: lambda grid at drop 60% ----
    RD60 = {sd: build_resid(z[sd], 0.60) for sd in SEEDS}
    log('\n[A] lambda_mono x lambda_conv  (L_flat=300,.002, drop 60%)')
    log(f'{"mono":>5} {"conv":>5} | {"RMSE":>12} | {"score":>6} | {"s9->s10":>11} | shape')
    best = None
    for lm in [0.5, 1, 2, 4, 6, 8]:
        for lc in [0.25, 0.5, 1, 2, 4]:
            r, rs, sc, s9, s10, ok = evaluate(RD60, lm, lc, 300.0, 0.002)
            log(f'{lm:>5.2g} {lc:>5.2g} | {r:>6.2f} ±{rs:.2f} | {sc:>6.1f} | {s9:>5.2f}->{s10:>4.2f} | {"PASS" if ok else "fail"}')
            if ok and (best is None or sc < best[2]): best = (lm, lc, sc, r)
    lm_b, lc_b = (best[0], best[1]) if best else (2.0, 0.25)
    log(f'--> A winner: mono={lm_b}, conv={lc_b}')

    # ---- B: flat grid at A winner ----
    log(f'\n[B] flat_w x margin  at (mono={lm_b}, conv={lc_b}), drop 60%')
    log(f'{"fw":>5} {"m":>6} | {"RMSE":>12} | {"score":>6} | {"s9->s10":>11} | shape')
    bestB = None
    for fw in [300, 500, 800, 1200]:
        for fm in [0.002, 0.004, 0.006]:
            r, rs, sc, s9, s10, ok = evaluate(RD60, lm_b, lc_b, float(fw), fm)
            log(f'{fw:>5} {fm:>6.3f} | {r:>6.2f} ±{rs:.2f} | {sc:>6.1f} | {s9:>5.2f}->{s10:>4.2f} | {"PASS" if ok else "fail"}')
            if ok and (bestB is None or sc < bestB[2]): bestB = (fw, fm, sc, r)
    fw_b, fm_b = (bestB[0], bestB[1]) if bestB else (300, 0.002)
    log(f'--> B winner: flat_w={fw_b}, m={fm_b}')

    # ---- C: threshold sweep at final HI ----
    log(f'\n[C] drop sweep 0-95%  at (mono={lm_b}, conv={lc_b}, flat={fw_b}, {fm_b})')
    log(f'{"drop":>5} | {"RMSE":>12} | {"score":>6}')
    bestC = None
    for dp in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]:
        RDd = {sd: build_resid(z[sd], dp) for sd in SEEDS}
        r, rs, sc, s9, s10, ok = evaluate(RDd, lm_b, lc_b, float(fw_b), fm_b)
        log(f'{dp:>5.0%} | {r:>6.2f} ±{rs:.2f} | {sc:>6.1f}')
        if bestC is None or sc < bestC[1]: bestC = (dp, sc, r, rs)
    log(f'--> C winner: drop={bestC[0]:.0%}  RMSE {bestC[2]:.2f} ±{bestC[3]:.2f}  score {bestC[1]:.1f}')
    log(f'\nFINAL (revised Stage 5): mono={lm_b} conv={lc_b} flat=({fw_b},{fm_b}) drop={bestC[0]:.0%}')
    log(f'reference (old Stage 5 official): 8.05 ±0.11 / 23.5 @ drop 85%')
    log(f'wall {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
