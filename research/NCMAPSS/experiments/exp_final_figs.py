"""
exp_final_figs.py — regenerate the PAPER FIGURE SET, everything computed under
the FINAL confirmed system:
  HI loss  (l_init, l_end, l_mono, l_conv) = (1, 1, 2, 0.25) + L_flat(300, 0.002)
  filter   NPER=200, drop 60%, PER-CYCLE causal detcov threshold
  GP       3 sensors, Matern3/2 r1, 8192 pts (cached stats)

Outputs:
  fig_final_hi.png               (a) test HI curves  (b) slope profile vs no-L_flat
  fig_final_lambda_heatmap.png   mono x conv grid @ final flat settings (3 panels)
  fig_final_threshold.png        drop sweep RMSE/score @ final HI
  fig_final_flatgrid.png         flat_w x margin heatmap (RMSE / score / tail margin)
Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_final_figs.py
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_hi_final_sweep import build, evaluate, SEEDS
from exp_ds03_port import rul_eval, nasa_score, FRACS
from neural_fusion_tail import train_model_tail

LM, LC, FW, FM = 2.0, 0.25, 300.0, 0.002       # FINAL config
TEAL, GRAY, RED = '#028090', '#808080', '#C0504D'
t0 = time.time()

RD = {sd: build(sd, 200) for sd in SEEDS}
print(f'residuals ready ({time.time()-t0:.0f}s)', flush=True)

def smooth1d(h, w=5):
    if len(h) < w: return np.asarray(h, float)
    return np.convolve(h, np.ones(w) / w, mode='valid')

def wslopes(h, nwin=10):
    hs = smooth1d(h); n = len(hs); x = np.linspace(0, 1, n)
    return np.array([np.polyfit(x[(x >= k/nwin) & (x <= (k+1)/nwin + 1e-9)],
                                hs[(x >= k/nwin) & (x <= (k+1)/nwin + 1e-9)], 1)[0] for k in range(nwin)])

def fit_hi(lm, lc, fw, fm, sd):
    dl, tl = RD[sd]
    np.random.seed(sd)
    him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                              lambda1=lm, lambda2=lc, init_threshold=0.2,
                              flat_w=fw, flat_m=fm, alpha=0.001)
    return him, dl, tl

# ---------- fig 1: HI curves + slope profile ----------
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7))
prof = {}
for name, fw, fm, color in [('final  L_flat(300, .002)', FW, FM, TEAL),
                            ('no L_flat (reference)', 0.0, 0.0, RED)]:
    slopes = []
    for sd in SEEDS:
        him, dl, tl = fit_hi(LM, LC, fw, fm, sd)
        for u, cy, d in tl:
            h = him.forward(d).flatten()
            slopes.append(wslopes(h))
            if sd == 0 and fw > 0:
                axes[0].plot(np.array(cy) / max(cy), h, lw=1.7, alpha=0.85, label=f'unit {u}')
    prof[name] = (np.nanmean(np.stack(slopes), 0), color)
ax = axes[0]
ax.axhline(1.0, color='r', ls='--', lw=1.2); ax.axhline(0.2, color='gray', ls=':', lw=1)
ax.set_ylim(0, 1.15); ax.grid(alpha=0.3)
ax.set_xlabel('life fraction'); ax.set_ylabel('Health Index')
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.set_title('(a) HI curves — final configuration (test units, seed 0)', fontsize=10.5)
ax = axes[1]
xd = np.arange(1, 11) * 10
for name, (S, color) in prof.items():
    ax.plot(xd, S, marker='o', ms=5, lw=2, color=color, label=name)
    ax.annotate(f'{S[9]:.2f}', (100, S[9]), textcoords='offset points', xytext=(8, -4),
                fontsize=9, color=color)
ax.axvspan(85, 100, color=RED, alpha=0.07)
ax.set_xlabel('life-fraction window (%)'); ax.set_ylabel('HI slope in window')
ax.grid(alpha=0.3); ax.legend(fontsize=9, loc='upper left')
ax.set_title('(b) slope profile — accelerating to the very end', fontsize=10.5)
fig.suptitle('Final HI — $\\lambda$=(1, 1, 2, 0.25) + $L_{flat}$(300, 0.002), drop 60% (causal)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(os.path.join(HERE, 'fig_final_hi.png'), dpi=170); plt.close(fig)
print(f'fig_final_hi.png saved ({time.time()-t0:.0f}s)', flush=True)

# ---------- fig 2: mono x conv heatmap @ final flat ----------
MONO = [0.5, 1, 2, 4, 6, 8]; CONV = [0.25, 0.5, 1, 2, 4]
R = np.zeros((6, 5)); SC = np.zeros((6, 5)); TM = np.zeros((6, 5))
for i, lm in enumerate(MONO):
    for j, lc in enumerate(CONV):
        r, rs, sc, st_, s9, s10, en, ok = evaluate(RD, lm, lc, FW, FM)
        R[i, j] = r; SC[i, j] = sc; TM[i, j] = s10 - s9
    print(f'  lambda grid row mono={lm} done ({time.time()-t0:.0f}s)', flush=True)
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
for ax, (M, title, cmap, fmt) in zip(axes, [
        (R, '(a) RUL RMSE (cycles)', 'viridis_r', '%.2f'),
        (SC, "(b) NASA PHM'08 score", 'viridis_r', '%.1f'),
        (TM, '(c) tail slope margin  s10 − s9\n(>0 = accelerating to the end)', 'RdYlGn', '%.2f')]):
    im = ax.imshow(M, cmap=cmap, aspect='auto')
    for i in range(6):
        for j in range(5):
            v = M[i, j]
            dark = im.norm(v) > 0.6 if cmap != 'RdYlGn' else abs(im.norm(v) - 0.5) > 0.32
            ax.text(j, i, fmt % v, ha='center', va='center', fontsize=8.5,
                    color='white' if dark else 'black')
    ax.set_xticks(range(5)); ax.set_xticklabels(CONV)
    ax.set_yticks(range(6)); ax.set_yticklabels(MONO)
    ax.set_xlabel('$\\lambda_{conv}$'); ax.set_ylabel('$\\lambda_{mono}$')
    ax.set_title(title, fontsize=10.5)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('Effect of ($\\lambda_{mono}$, $\\lambda_{conv}$) under the final system — $L_{flat}$(300, .002) ON, 3 seeds', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(HERE, 'fig_final_lambda_heatmap.png'), dpi=180); plt.close(fig)
print(f'fig_final_lambda_heatmap.png saved ({time.time()-t0:.0f}s)', flush=True)

# ---------- fig 3: drop sweep @ final HI ----------
DROPS = [0.0, 0.10, 0.25, 0.35, 0.50, 0.60, 0.70, 0.80]
dr_r, dr_rs, dr_s, dr_ss = [], [], [], []
for dp in DROPS:
    RDd = {sd: build(sd, 200, drop=dp) for sd in SEEDS}
    rmses, scores = [], []
    for sd in SEEDS:
        dl, tl = RDd[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([d for _, _, d in dl], epochs=1000, lambda0=1.0,
                                  lambda1=LM, lambda2=LC, init_threshold=0.2,
                                  flat_w=FW, flat_m=FM, alpha=0.001)
        dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
        tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
        P, T, FR, UN = rul_eval(dHI, tHI, FRACS)
        rmses.append(float(np.sqrt(((P - T) ** 2).mean()))); scores.append(nasa_score(P, T))
    dr_r.append(np.mean(rmses)); dr_rs.append(np.std(rmses))
    dr_s.append(np.mean(scores)); dr_ss.append(np.std(scores))
    print(f'  drop {dp:.0%}: {dr_r[-1]:.2f} / {dr_s[-1]:.1f} ({time.time()-t0:.0f}s)', flush=True)
xd = [d * 100 for d in DROPS]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, (y, ys, lab, best) in zip(axes, [(dr_r, dr_rs, 'RUL RMSE (cycles)', None),
                                         (dr_s, dr_ss, "NASA PHM'08 score", None)]):
    ax.errorbar(xd, y, yerr=ys, color=TEAL, lw=2, marker='o', ms=5, capsize=3)
    k = int(np.argmin(y))
    ax.scatter([xd[k]], [y[k]], s=140, facecolors='none', edgecolors=RED, lw=2, zorder=5,
               label=f'optimum: drop {xd[k]:.0f}%')
    ax.axvspan(50, 70, color=TEAL, alpha=0.08)
    ax.set_xlabel('drop ratio (%) — per-cycle causal detcov percentile')
    ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=9)
axes[0].set_title('(a) RMSE vs filter threshold', fontsize=10.5)
axes[1].set_title('(b) NASA score vs filter threshold', fontsize=10.5)
fig.suptitle('Uncertainty-filter threshold sweep under the FINAL HI (3 seeds, mean ± std)', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(os.path.join(HERE, 'fig_final_threshold.png'), dpi=180); plt.close(fig)
print(f'fig_final_threshold.png saved ({time.time()-t0:.0f}s)', flush=True)

# ---------- fig 4: flat_w x margin heatmap (stage-B data) ----------
FWs = [300, 500, 800, 1200]; FMs = [0.002, 0.004, 0.006]
RB = np.array([[8.20, 8.26, 8.35], [8.25, 8.34, 8.45], [8.30, 8.42, 8.57], [8.38, 8.51, 8.70]])
SB = np.array([[24.0, 24.6, 25.5], [24.4, 25.2, 26.3], [24.9, 26.0, 27.1], [25.5, 26.7, 28.1]])
TB = np.array([[.08, .17, .27], [.16, .25, .34], [.21, .30, .39], [.26, .35, .44]])
fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.4))
for ax, (M, title, cmap, fmt) in zip(axes, [
        (RB, '(a) RUL RMSE', 'viridis_r', '%.2f'), (SB, '(b) NASA score', 'viridis_r', '%.1f'),
        (TB, '(c) tail margin s10−s9', 'Greens', '%.2f')]):
    im = ax.imshow(M, cmap=cmap, aspect='auto')
    for i in range(4):
        for j in range(3):
            ax.text(j, i, fmt % M[i, j], ha='center', va='center', fontsize=9,
                    color='white' if im.norm(M[i, j]) > 0.6 else 'black')
    ax.set_xticks(range(3)); ax.set_xticklabels(FMs)
    ax.set_yticks(range(4)); ax.set_yticklabels(FWs)
    ax.set_xlabel('margin  m'); ax.set_ylabel('$w_{flat}$')
    ax.set_title(title, fontsize=10.5)
    plt.colorbar(im, ax=ax, fraction=0.046)
fig.suptitle('Effect of $L_{flat}$ strength ($w_{flat}$, margin) at $\\lambda$=(2, 0.25) — all shape-PASS', fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.88])
fig.savefig(os.path.join(HERE, 'fig_final_flatgrid.png'), dpi=180); plt.close(fig)
print(f'fig_final_flatgrid.png saved ({time.time()-t0:.0f}s)', flush=True)
print('ALL FIGURES DONE')
