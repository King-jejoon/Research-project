"""
exp_hi_final_sweep.py — RE-SWEEP all HI parameters under the FINAL loss system
(L_flat active) with the deployment-correct per-cycle causal detcov threshold.

  A) lambda_mono x lambda_conv grid  WITH L_flat(800, .004) ON
  B) flat_w x margin grid at the stage-A winner
  C) NPER 60 vs 200 under the final HI

Selection rule: shape first (tail s10>=s9, start<=0.30, end>=0.93),
then lowest NASA score among shape-passing.
Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_final_sweep.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_ds03_port import rul_eval, nasa_score, FRACS
from neural_fusion_tail import train_model_tail

SENS = ['T48', 'T50', 'Wf']
SEEDS = [0, 1, 2]
DROP = 0.60
RES = os.path.join(HERE, 'hi_final_sweep_results.txt')
CACHES = {200: 'port_stats_s3_matern32_r1_m18_n8192_p200_s{sd}.npz',
          60:  'port_stats_s3_matern32_r1_m18_n8192_s{sd}.npz'}


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def build(sd, nper=200, drop=DROP):
    """residuals with per-cycle CAUSAL detcov threshold."""
    z = np.load(os.path.join(HERE, CACHES[nper].format(sd=sd)))
    du = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('dev_') and k.endswith('_cc')})
    tu = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('test_') and k.endswith('_cc')})
    def rd(units, split):
        out = {}
        for u in units:
            key = f'{split}_{u}'; cc = z[f'{key}_cc']; r = z[f'{key}_resid']; dc = z[f'{key}_detcov']
            d = {s: {} for s in SENS}
            for k in np.unique(cc):
                m = cc == k
                if drop > 0:
                    thr = np.percentile(dc[m], drop * 100)
                    keep = m & (dc >= thr)
                    if keep.sum() == 0: keep = m
                else:
                    keep = m
                mv = r[keep].mean(0)
                for j, s in enumerate(SENS): d[s][int(k)] = float(mv[j])
            out[u] = d
        return out
    dev, test = rd(du, 'dev'), rd(tu, 'test')
    st = {s: (np.mean([v for u in dev for v in dev[u][s].values()]),
              np.std([v for u in dev for v in dev[u][s].values()]) + 1e-8) for s in SENS}
    nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
    dn, tn = nrm(dev), nrm(test)
    bl = lambda dd: [(u, sorted(dd[u][SENS[0]].keys()),
                      np.array([[dd[u][s][k] for s in SENS] for k in sorted(dd[u][SENS[0]].keys())]))
                     for u in sorted(dd)]
    return bl(dn), bl(tn)


def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def shape_of(h):
    hs = smooth1d(h); n = len(hs); x = np.linspace(0, 1, n)
    start = hs[x <= 0.10].mean() if (x <= 0.10).any() else hs[0]
    s9m = (x >= 0.8) & (x <= 0.9); s10m = (x >= 0.9)
    s9 = np.polyfit(x[s9m], hs[s9m], 1)[0]; s10 = np.polyfit(x[s10m], hs[s10m], 1)[0]
    return start, s9, s10, hs[-1]


def evaluate(RD_by_seed, lm, lc, fw, fm):
    rmses, scores, mets = [], [], []
    for sd in SEEDS:
        dl, tl = RD_by_seed[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                  lambda1=lm, lambda2=lc, init_threshold=0.2,
                                  flat_w=fw, flat_m=fm, alpha=0.001)
        dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
        tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
        P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
        rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
        scores.append(nasa_score(P, T))
        for u in tHI: mets.append(shape_of(tHI[u]))
    start, s9, s10, end = np.nanmean(np.array(mets), 0)
    ok = (s10 >= s9) and (start <= 0.30) and (end >= 0.93)
    return (np.mean(rmses), np.std(rmses), np.mean(scores), start, s9, s10, end, ok)


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('FINAL-SYSTEM HI RE-SWEEP  (L_flat ON, per-cycle causal threshold, drop 60%)')
    RD = {sd: build(sd, 200) for sd in SEEDS}
    log(f'residuals ready ({time.time()-t0:.0f}s)')

    # ---- A: mono x conv grid with L_flat(800, .004) ----
    log('\n[A] lambda_mono x lambda_conv  (L_flat=800, m=.004)')
    log(f'{"mono":>5} {"conv":>5} | {"RMSE":>11} | {"score":>5} | {"s9":>5} {"s10":>5} {"start":>5} | shape')
    best = None
    for lm in [0.5, 1, 2, 4, 6, 8]:
        for lc in [0.25, 0.5, 1, 2, 4]:
            r, rs, sc, st_, s9, s10, en, ok = evaluate(RD, lm, lc, 800.0, 0.004)
            log(f'{lm:>5.2g} {lc:>5.2g} | {r:>5.2f} ±{rs:.2f} | {sc:>5.1f} | {s9:>5.2f} {s10:>5.2f} {st_:>5.2f} | {"PASS" if ok else "fail"}')
            if ok and (best is None or sc < best[2]):
                best = (lm, lc, sc, r)
    lm_b, lc_b = (best[0], best[1]) if best else (2.0, 0.5)
    log(f'--> stage-A winner: mono={lm_b}, conv={lc_b}')

    # ---- B: flat_w x margin at the winner ----
    log(f'\n[B] flat_w x margin  at (mono={lm_b}, conv={lc_b})')
    log(f'{"fw":>5} {"m":>6} | {"RMSE":>11} | {"score":>5} | {"s9":>5} {"s10":>5} | shape')
    bestB = None
    for fw in [300, 500, 800, 1200]:
        for fm in [0.002, 0.004, 0.006]:
            r, rs, sc, st_, s9, s10, en, ok = evaluate(RD, lm_b, lc_b, float(fw), fm)
            log(f'{fw:>5} {fm:>6.3f} | {r:>5.2f} ±{rs:.2f} | {sc:>5.1f} | {s9:>5.2f} {s10:>5.2f} | {"PASS" if ok else "fail"}')
            if ok and (bestB is None or sc < bestB[2]):
                bestB = (fw, fm, sc, r)
    fw_b, fm_b = (bestB[0], bestB[1]) if bestB else (800, 0.004)
    log(f'--> stage-B winner: flat_w={fw_b}, m={fm_b}')

    # ---- C: NPER check under final HI ----
    log(f'\n[C] NPER 60 vs 200  at (mono={lm_b}, conv={lc_b}, flat={fw_b}, m={fm_b})')
    for nper in [60, 200]:
        RDn = {sd: build(sd, nper) for sd in SEEDS}
        r, rs, sc, st_, s9, s10, en, ok = evaluate(RDn, lm_b, lc_b, float(fw_b), fm_b)
        log(f'  NPER={nper:>3}: RMSE {r:.2f} ±{rs:.2f}  score {sc:.1f}  s9->s10 {s9:.2f}->{s10:.2f}  {"PASS" if ok else "fail"}')
    log(f'\nwall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
