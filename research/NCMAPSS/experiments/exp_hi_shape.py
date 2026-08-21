"""
exp_hi_shape.py — HI coefficient grid sweep scored by CURVE SHAPE first,
RMSE/score second.  Shape requirements (domain criteria):

  1. start low        : mean HI over first 10% of life <= 0.35
                        (rules out the all-above-0.9 overfit mode)
  2. not a straight line / slope must GROW: segment slopes s1 < s2 < s3
                        (thirds of life fraction, linear fit each)
  3. NO end-flattening: s3/s2 > 1 (bigger = better); s3/s2 < 1 is the
                        overfit signature (rises then goes flat at the end)
  4. reaches failure  : final HI >= 0.90

Grid: l_mono x l_conv (l_init=1, thr=0.2 fixed — shown insensitive).
Curves: test units, smoothed (moving average w=5) before slope estimation.
Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_shape.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_hi_sweep import build_residuals
from exp_ds03_port import rul_eval, nasa_score, FRACS
import neural_fusion as NF

SEEDS = [0, 1, 2]
MONO = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0]
CONV = [0.25, 0.5, 1.0, 2.0, 4.0]
RES = os.path.join(HERE, 'hi_shape_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def smooth(h, w=5):
    if len(h) < w: return h
    k = np.ones(w) / w
    return np.convolve(h, k, mode='valid')


def shape_metrics(h):
    """h: one HI curve. Returns start, first-half mean, s1,s2,s3, end, d9, d10.
    d9/d10 = decile slopes of the last two 10% windows (end-flattening check)."""
    hs = smooth(np.asarray(h, float))
    n = len(hs); x = np.linspace(0, 1, n)
    start = hs[x <= 0.10].mean() if (x <= 0.10).any() else hs[0]
    half = hs[x <= 0.5].mean()
    end = hs[-1]
    sl = []
    for a, b in [(0.0, 1/3), (1/3, 2/3), (2/3, 1.0)]:
        m = (x >= a) & (x <= b)
        sl.append(np.polyfit(x[m], hs[m], 1)[0] if m.sum() > 2 else np.nan)
    dec = []
    for a, b in [(0.8, 0.9), (0.9, 1.0)]:
        m = (x >= a) & (x <= b + 1e-9)
        dec.append(np.polyfit(x[m], hs[m], 1)[0] if m.sum() > 2 else np.nan)
    return start, half, sl[0], sl[1], sl[2], end, dec[0], dec[1]


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('HI SHAPE-FIRST GRID v2  (l_init=1, thr=0.2 fixed;  3 seeds; test curves)')
    log('pass rule: start<=0.35 & s1<s2<s3 & end>=0.90 & NO END-FLATTENING (d10>=d9, last two deciles)')
    RD = {sd: build_residuals(sd) for sd in SEEDS}
    log(f'residuals ready ({time.time()-t0:.0f}s)\n')
    hdr = (f'{"mono":>5} {"conv":>5} | {"RMSE":>5} {"score":>5} | '
           f'{"start":>5} {"s1":>5} {"s2":>5} {"s3":>5} {"d9":>5} {"d10":>5} {"end":>5} | verdict')
    log(hdr); log('-' * len(hdr))
    results = []
    for lm in MONO:
        for lc in CONV:
            rmses, scores, mets = [], [], []
            for sd in SEEDS:
                dl, tl = RD[sd]
                np.random.seed(sd)
                him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                        lambda1=lm, lambda2=lc, init_threshold=0.2,
                                        alpha=0.001, verbose=False)
                dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
                tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
                P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                scores.append(nasa_score(P, T))
                for u in tHI: mets.append(shape_metrics(tHI[u]))
            rmse = np.mean(rmses); score = np.mean(scores)
            start, half, s1, s2, s3, end, d9, d10 = np.nanmean(np.array(mets), 0)
            ok = (start <= 0.35) and (s1 < s2 < s3) and (end >= 0.90) and (d10 >= d9)
            verdict = 'PASS' if ok else ('fail(END-FLAT)' if d10 < d9 else 'fail')
            log(f'{lm:>5.2g} {lc:>5.2g} | {rmse:>5.2f} {score:>5.1f} | '
                f'{start:>5.2f} {s1:>5.2f} {s2:>5.2f} {s3:>5.2f} {d9:>5.2f} {d10:>5.2f} {end:>5.2f} | {verdict}')
            results.append(dict(lm=lm, lc=lc, rmse=rmse, score=score, start=start,
                                s1=s1, s2=s2, s3=s3, d9=d9, d10=d10, end=end, ok=ok))
    passed = [r for r in results if r['ok']]
    log('\n' + '=' * 70)
    if passed:
        best = sorted(passed, key=lambda r: r['score'])[:6]
        log('TOP among SHAPE-PASSING configs (no end-flattening), sorted by score:')
        for r in best:
            log(f'  mono={r["lm"]:g} conv={r["lc"]:g}: RMSE {r["rmse"]:.2f}  score {r["score"]:.1f}  '
                f'd9->d10 {r["d9"]:.2f}->{r["d10"]:.2f}  start={r["start"]:.2f}')
    else:
        log('NO config passed all shape rules — inspect table above.')
    log(f'wall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
