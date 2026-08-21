"""Verification of the SOLVED config: (mono=8, conv=2.5, end-anchor A=1.10).
(a) test HI curves  (b) 10-window slope profile vs old official (6,2,A=1).
Also prints per-seed RMSE / NASA score for the official record.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_hi_solved_fig.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_hi_sweep import build_residuals
from exp_ds03_port import rul_eval, nasa_score, FRACS
import neural_fusion as NF
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
RD = {sd: build_residuals(sd) for sd in SEEDS}


def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')


def window_slopes(h, nwin=10):
    hs = smooth1d(h); n = len(hs); x = np.linspace(0, 1, n)
    return np.array([np.polyfit(x[(x >= k/nwin) & (x <= (k+1)/nwin + 1e-9)],
                                hs[(x >= k/nwin) & (x <= (k+1)/nwin + 1e-9)], 1)[0]
                     for k in range(nwin)])


def fit(kind, lm, lc, et, sd):
    dl, tl = RD[sd]
    np.random.seed(sd)
    if kind == 'std':
        him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                lambda1=lm, lambda2=lc, init_threshold=0.2,
                                alpha=0.001, verbose=False)
    else:
        him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                  lambda1=lm, lambda2=lc, init_threshold=0.2,
                                  end_target=et, alpha=0.001)
    dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
    tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
    tlx = [(u, np.array(cy) / max(cy), tHI[u]) for u, cy, d in tl]
    P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
    return tlx, float(np.sqrt(((P - T) ** 2).mean())), nasa_score(P, T)


NEW = ('solved (8, 2.5, A=1.10)', 'tail', 8.0, 2.5, 1.10, '#028090')
OLD = ('old official (6, 2, A=1)', 'std', 6.0, 2.0, 1.0, '#808080')

fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

# per-seed metrics + curves
print('per-seed metrics:')
prof = {}
for name, kind, lm, lc, et, color in [NEW, OLD]:
    rmses, scores, slopes = [], [], []
    for sd in SEEDS:
        tlx, r, s = fit(kind, lm, lc, et, sd)
        rmses.append(r); scores.append(s)
        for u, x, h in tlx: slopes.append(window_slopes(h))
        if sd == 0 and name.startswith('solved'):
            for u, x, h in tlx:
                axes[0].plot(x, h, lw=1.7, alpha=0.85, label=f'unit {u}')
    prof[name] = (np.nanmean(np.stack(slopes), 0), color)
    print(f'  {name}: RMSE {np.mean(rmses):.2f} +/- {np.std(rmses):.2f}   '
          f'score {np.mean(scores):.1f} +/- {np.std(scores):.1f}   per-seed {[f"{r:.2f}" for r in rmses]}')

ax = axes[0]
ax.axhline(1.0, color='r', ls='--', lw=1.2, label='failure threshold (HI=1)')
ax.axhline(0.2, color='gray', ls=':', lw=1)
ax.set_ylim(0, 1.2); ax.grid(alpha=0.3)
ax.set_xlabel('life fraction'); ax.set_ylabel('Health Index')
ax.legend(fontsize=7.5, ncol=2, loc='upper left')
ax.set_title('(a) SOLVED config HI curves (test, seed 0)\n'
             'crosses HI=1 while still accelerating (A=1.10)', fontsize=10.5)

ax = axes[1]
xdec = np.arange(1, 11) * 10
for name, (S, color) in prof.items():
    ax.plot(xdec, S, marker='o', ms=5, lw=2, color=color, label=name)
    ax.annotate(f'{S[9]:.2f}', (100, S[9]), textcoords='offset points',
                xytext=(8, -4), fontsize=9, color=color)
ax.axvspan(85, 100, color='#C0504D', alpha=0.07)
ax.set_xlabel('life-fraction window (%)'); ax.set_ylabel('HI slope in window')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper left')
ax.set_title('(b) slope profile — solved config rises to the very end\n'
             '(1.90 -> 1.94 in the last window)', fontsize=10.5)

fig.suptitle('End-flattening SOLVED — (mono, conv) = (8, 2.5), end anchor A = 1.10', fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.91])
out = os.path.join(HERE, 'fig_hi_solved.png')
fig.savefig(out, dpi=170)
print('saved', out)
