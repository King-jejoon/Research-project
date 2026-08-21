"""Fine-grained slope check: is the HI slope increasing all the way to the end?
Splits each test HI curve into 10 deciles of life fraction, fits a line in each,
and prints the 10 slopes.  End-flattening = slope of decile 10 < decile 9.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_hi_slope_check.py
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_hi_sweep import build_residuals
import neural_fusion as NF

SEEDS = [0, 1, 2]
CANDS = [
    ('A: mono=1,  conv=0.5 ', 1.0, 0.5),
    ('B: mono=2,  conv=0.5 ', 2.0, 0.5),
    ('C: mono=2,  conv=0.25', 2.0, 0.25),
    ('D: mono=1,  conv=4   ', 1.0, 4.0),
    ('E: mono=0.5, conv=8  ', 0.5, 8.0),
    ('prev: mono=6, conv=2 ', 6.0, 2.0),
]


def smooth(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def decile_slopes(h):
    hs = smooth(h); n = len(hs); x = np.linspace(0, 1, n)
    out = []
    for k in range(10):
        a, b = k / 10, (k + 1) / 10
        m = (x >= a) & (x <= b + 1e-9)
        out.append(np.polyfit(x[m], hs[m], 1)[0] if m.sum() > 2 else np.nan)
    return np.array(out)


RD = {sd: build_residuals(sd) for sd in SEEDS}
print('decile slopes of test HI curves (mean over 6 units x 3 seeds)')
print('columns = life-fraction windows;  END-FLAT if slope[10] < slope[9]\n')
hdr = 'config                 | ' + ' '.join(f'{(k+1)*10:>5d}%' for k in range(10)) + ' | end-flat?'
print(hdr); print('-' * len(hdr))
for name, lm, lc in CANDS:
    allsl = []
    for sd in SEEDS:
        dl, tl = RD[sd]
        np.random.seed(sd)
        him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                lambda1=lm, lambda2=lc, init_threshold=0.2,
                                alpha=0.001, verbose=False)
        for u, cy, d in tl:
            allsl.append(decile_slopes(him.forward(d).flatten()))
    S = np.nanmean(np.stack(allsl), 0)
    flat = 'YES <-- ' if S[9] < S[8] else 'no'
    print(f'{name} | ' + ' '.join(f'{v:>6.2f}' for v in S) + f' | {flat}')

# per-unit detail for the currently recommended candidate B (seed 0)
print('\nper-unit decile slopes — candidate B (mono=2, conv=0.5), seed 0')
dl, tl = RD[0]
np.random.seed(0)
him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                        lambda1=2.0, lambda2=0.5, init_threshold=0.2,
                        alpha=0.001, verbose=False)
print('unit | ' + ' '.join(f'{(k+1)*10:>5d}%' for k in range(10)) + ' | end-flat?')
for u, cy, d in tl:
    S = decile_slopes(him.forward(d).flatten())
    flat = 'YES' if S[9] < S[8] else 'no'
    print(f'{u:>4} | ' + ' '.join(f'{v:>6.2f}' for v in S) + f' | {flat}')
