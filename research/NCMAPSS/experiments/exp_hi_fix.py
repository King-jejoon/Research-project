"""
exp_hi_fix.py — fix the end-flattening of the low-RMSE HI region.

Goal: RMSE ~7.9-8.0 AND no end-flattening (d10 >= d9).
  A. boundary fine grid: conv 2.5-3.5 at low mono; very high mono at low conv;
     fewer epochs (overfitting develops late in training).
  B. TAIL-WEIGHTED convexity (neural_fusion_tail): keep global conv low for
     RMSE, ramp the convexity penalty toward end of life to force tail
     acceleration:  w_t = 1 + beta*(t/T)^2.

Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_fix.py
"""
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_hi_sweep import build_residuals
from exp_ds03_port import rul_eval, nasa_score, FRACS
import neural_fusion as NF
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'hi_fix_results.txt')

# (name, kind, params)
CONFIGS = [
    ('ref  (6,2)          ', 'std',  dict(lm=6.0, lc=2.0, ep=1000)),
    ('ref  (2,0.5) endflat', 'std',  dict(lm=2.0, lc=0.5, ep=1000)),
    # A1: boundary fine grid
    ('A (0.5,2.5)         ', 'std',  dict(lm=0.5, lc=2.5, ep=1000)),
    ('A (1,2.5)           ', 'std',  dict(lm=1.0, lc=2.5, ep=1000)),
    ('A (2,2.5)           ', 'std',  dict(lm=2.0, lc=2.5, ep=1000)),
    ('A (0.5,3)           ', 'std',  dict(lm=0.5, lc=3.0, ep=1000)),
    ('A (1,3)             ', 'std',  dict(lm=1.0, lc=3.0, ep=1000)),
    ('A (2,3)             ', 'std',  dict(lm=2.0, lc=3.0, ep=1000)),
    ('A (10,0.5)          ', 'std',  dict(lm=10.0, lc=0.5, ep=1000)),
    ('A (12,1)            ', 'std',  dict(lm=12.0, lc=1.0, ep=1000)),
    # A2: fewer epochs at the low-RMSE configs (late-training overfit check)
    ('A ep500  (2,0.5)    ', 'std',  dict(lm=2.0, lc=0.5, ep=500)),
    ('A ep700  (2,0.5)    ', 'std',  dict(lm=2.0, lc=0.5, ep=700)),
    ('A ep500  (1,1)      ', 'std',  dict(lm=1.0, lc=1.0, ep=500)),
    # B: tail-weighted convexity  (low global conv + strong tail ramp)
    ('B (2,0.5) beta=6    ', 'tail', dict(lm=2.0, lc=0.5, ep=1000, beta=6.0)),
    ('B (2,0.5) beta=12   ', 'tail', dict(lm=2.0, lc=0.5, ep=1000, beta=12.0)),
    ('B (2,0.5) beta=24   ', 'tail', dict(lm=2.0, lc=0.5, ep=1000, beta=24.0)),
    ('B (1,0.5) beta=12   ', 'tail', dict(lm=1.0, lc=0.5, ep=1000, beta=12.0)),
    ('B (2,1)   beta=8    ', 'tail', dict(lm=2.0, lc=1.0, ep=1000, beta=8.0)),
]


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def smooth(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def last_deciles(h):
    hs = smooth(h); n = len(hs); x = np.linspace(0, 1, n)
    out = []
    for a, b in [(0.8, 0.9), (0.9, 1.0)]:
        m = (x >= a) & (x <= b + 1e-9)
        out.append(np.polyfit(x[m], hs[m], 1)[0] if m.sum() > 2 else np.nan)
    start = hs[x <= 0.10].mean() if (x <= 0.10).any() else hs[0]
    return start, out[0], out[1], hs[-1]


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('HI END-FLATTENING FIX  (goal: RMSE<8.1 AND d10>=d9)  3 seeds')
    RD = {sd: build_residuals(sd) for sd in SEEDS}
    hdr = f'{"config":>22} | {"RMSE":>5} {"score":>5} | {"start":>5} {"d9":>5} {"d10":>5} {"end":>5} | verdict'
    log(hdr); log('-' * len(hdr))
    for name, kind, c in CONFIGS:
        rmses, scores, mets = [], [], []
        for sd in SEEDS:
            dl, tl = RD[sd]
            np.random.seed(sd)
            if kind == 'std':
                him, _ = NF.train_model([d for _, _, d in dl], epochs=c['ep'], lambda0=1.0,
                                        lambda1=c['lm'], lambda2=c['lc'], init_threshold=0.2,
                                        alpha=0.001, verbose=False)
            else:
                him, _ = train_model_tail([d for _, _, d in dl], epochs=c['ep'], lambda0=1.0,
                                          lambda1=c['lm'], lambda2=c['lc'], init_threshold=0.2,
                                          tail_beta=c['beta'], gamma=2.0, alpha=0.001)
            dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
            tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
            P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
            rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
            scores.append(nasa_score(P, T))
            for u in tHI: mets.append(last_deciles(tHI[u]))
        rmse = np.mean(rmses); score = np.mean(scores)
        start, d9, d10, end = np.nanmean(np.array(mets), 0)
        ok = (d10 >= d9) and (start <= 0.35) and (end >= 0.90)
        verdict = ('PASS' if ok else ('END-FLAT' if d10 < d9 else 'fail'))
        mark = '  <== TARGET' if ok and rmse < 8.1 else ''
        log(f'{name:>22} | {rmse:>5.2f} {score:>5.1f} | {start:>5.2f} {d9:>5.2f} {d10:>5.2f} {end:>5.2f} | {verdict}{mark}')
    log(f'wall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
