"""
exp_hi_ideal.py — SOLVE THE END-FLATTENING (last-window slope must be max).

Residual input = existing pipeline output (NPER=200, detcov drop 60%) — no
extra smoothing (that role already belongs to the uncertainty filter).

Two levers aimed directly at the tail:
  1) longer training: tail acceleration grows with epochs (ep500 collapse
     showed the direction) -> ep 2000/3000/5000 may rescue low-conv configs.
  2) CEILING term (neural_fusion_tail): before ceil_frac of life, penalize
     HI > cmax -> the curve cannot arrive early at 1, forcing the steep
     final rise.  Attacks the flattening mechanism itself.

Full 10-window slope profile printed for every config.
verdict: endflat if s10 < s9;  IDEAL if every window rises (eps 0.05),
start<=0.30, end>=0.93.
Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_ideal.py
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
EPS = 0.05
RES = os.path.join(HERE, 'hi_ideal_results.txt')

CONFIGS = [
    # round 6 — METHOD 1: window-level anti-flattening loss L_flat (end-only,
    # macro).  h(T)=1 kept (end_target=1).  Shape-first evaluation.
    ('ref (6,2)               ', 'std',  dict(lm=6, lc=2, ep=1000)),
    ('ref (2,0.5)             ', 'std',  dict(lm=2, lc=0.5, ep=1000)),
    ('(2,0.5) flat300         ', 'tail', dict(lm=2, lc=0.5, ep=1000, cw=0.0, cmax=0.92, fw=300.0, fm=0.0)),
    ('(2,0.5) flat300 m=.002  ', 'tail', dict(lm=2, lc=0.5, ep=1000, cw=0.0, cmax=0.92, fw=300.0, fm=0.002)),
    ('(2,0.5) flat300 m=.004  ', 'tail', dict(lm=2, lc=0.5, ep=1000, cw=0.0, cmax=0.92, fw=300.0, fm=0.004)),
    ('(2,0.5) flat500 m=.002  ', 'tail', dict(lm=2, lc=0.5, ep=1000, cw=0.0, cmax=0.92, fw=500.0, fm=0.002)),
    ('(2,0.5) flat800 m=.004  ', 'tail', dict(lm=2, lc=0.5, ep=1000, cw=0.0, cmax=0.92, fw=800.0, fm=0.004)),
    ('(2,2)   flat300 m=.002  ', 'tail', dict(lm=2, lc=2, ep=1000, cw=0.0, cmax=0.92, fw=300.0, fm=0.002)),
]


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def window_slopes(h, nwin=10):
    hs = smooth1d(h); n = len(hs); x = np.linspace(0, 1, n)
    out = []
    for k in range(nwin):
        a, b = k / nwin, (k + 1) / nwin
        m = (x >= a) & (x <= b + 1e-9)
        out.append(np.polyfit(x[m], hs[m], 1)[0] if m.sum() > 2 else np.nan)
    start = hs[x <= 0.10].mean() if (x <= 0.10).any() else hs[0]
    return np.array(out), start, hs[-1]


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log(f'HI END-FLATTENING SOLVE  (input = drop-60% residuals as-is; EPS={EPS})')
    RD = {sd: build_residuals(sd) for sd in SEEDS}
    log(f'{"config":>26} | {"RMSE":>5} {"scr":>5} | slopes 10%%..100%%')
    log('-' * 120)
    for name, kind, c in CONFIGS:
        rmses, scores, slopes, starts, ends = [], [], [], [], []
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
                                          ceil_w=c['cw'], cmax=c['cmax'], ceil_frac=0.90,
                                          end_target=c.get('et', 1.0),
                                          flat_w=c.get('fw', 0.0), flat_m=c.get('fm', 0.0),
                                          alpha=0.001)
            dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
            tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
            P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
            rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
            scores.append(nasa_score(P, T))
            for u in tHI:
                S, st, en = window_slopes(tHI[u])
                slopes.append(S); starts.append(st); ends.append(en)
        rmse = np.mean(rmses); score = np.mean(scores)
        S = np.nanmean(np.stack(slopes), 0); start = np.mean(starts); end = np.mean(ends)
        endok = S[9] >= S[8]
        rises = all(S[k+1] >= S[k] - EPS for k in range(9))
        ok = rises and endok and (start <= 0.30) and (end >= 0.93)
        v = 'IDEAL' if ok else ('endflat' if not endok else ('dip' if not rises else 'fail'))
        log(f'{name:>26} | {rmse:>5.2f} {score:>5.1f} | ' +
            ' '.join(f'{x:>5.2f}' for x in S) + f' | st{start:.2f} en{end:.2f} | {v}')
    log(f'wall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
