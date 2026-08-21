"""robust 검증 요약 그림. 값은 exp_robust.py / exp_robust_hi.py 실행 결과."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
fracs = [0, 10, 20, 30]

# Stage A: clean held-out RMSE (출력 스파이크 주입)
A = {
    'baseline (linspace)':   [0.91, 27.18, 33.75, 41.06],
    'A-1 window median':     [0.94, 1.08, 1.13, 1.36],
    'A-2 Hampel+linspace':   [0.96, 12.35, 22.77, 28.35],
}
# Stage B: trajectory deviation from clean (잔차 이상치 주입)
B = {
    'mean (baseline)':  [0.0, 2.08, 3.30, 4.14],
    'median':           [0.0, 0.09, 0.26, 0.58],
    'MAD-reject':       [0.0, 0.08, 0.31, 1.01],
    'trimmed 20%':      [0.0, 0.22, 0.98, 2.41],
}

fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

mk = {'baseline (linspace)': 'o-', 'A-1 window median': 's-', 'A-2 Hampel+linspace': '^-'}
for name, ys in A.items():
    ax[0].plot(fracs, ys, mk.get(name, 'o-'), lw=2, ms=7, label=name)
ax[0].set_yscale('log')
ax[0].set_title('Stage A — training-point selection\n(robust methods resist spike contamination)')
ax[0].set_xlabel('contamination fraction (%)'); ax[0].set_ylabel('held-out healthy RMSE (log)')
ax[0].legend(); ax[0].grid(alpha=0.3, which='both')
ax[0].annotate('baseline collapses\n(0.9 -> 41)', xy=(30, 41), xytext=(12, 20),
               arrowprops=dict(arrowstyle='->', color='tab:red'), color='tab:red')

for name, ys in B.items():
    ax[1].plot(fracs, ys, 'o-', lw=2, ms=6, label=name)
ax[1].set_title('Stage B — residual cycle-aggregation\n(robust stats keep trajectory near clean)')
ax[1].set_xlabel('contamination fraction (%)'); ax[1].set_ylabel('deviation from clean trajectory')
ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].text(2, 3.6, 'HI trend rho @20%:\n mean 0.69 (broken)\n median 0.97 (kept)',
           fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow'))

plt.tight_layout()
out = os.path.join(HERE, 'fig_robust_summary.png')
plt.savefig(out, dpi=110); plt.close()
print('saved ->', out)


if __name__ == '__main__':
    pass
