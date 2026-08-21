"""Visualize the detcov uncertainty filter (drop 60%) on residual data.
One separate figure per (unit, sensor) — 6 files total, narrow x-axis so the
degradation trend reads strongly.  Red = removed points (bottom 60% detcov),
blue = kept (top 40%).  Lines: cycle-mean before (gray dashed) / after (teal).
Run: /opt/anaconda3/envs/pt_prac/bin/python3 make_filter_plot.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial Unicode MS'
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, 'port_stats_s3_matern32_r1_m18_n8192_p200_s0.npz'))
SENS = ['T48', 'T50', 'Wf']
DROP = 0.60
UNITS = [('dev', 5), ('test', 10)]

for split, u in UNITS:
    cc = z[f'{split}_{u}_cc']; r = z[f'{split}_{u}_resid']; dc = z[f'{split}_{u}_detcov']
    thr = np.percentile(dc, DROP * 100)
    keep = dc >= thr
    ucyc = np.unique(cc)
    tru = None
    if split == 'dev':
        hs = z[f'{split}_{u}_hs']
        below = np.where(hs < 0.5)[0]
        if len(below): tru = ucyc[below[0]]
    for i, s in enumerate(SENS):
        fig, ax = plt.subplots(figsize=(6.0, 4.8))
        ax.scatter(cc[~keep], r[~keep, i], s=5, c='crimson', alpha=0.30,
                   label=f'剔除点 (detcov 最低 {DROP:.0%})', rasterized=True)
        ax.scatter(cc[keep], r[keep, i], s=5, c='#2f5c8f', alpha=0.22,
                   label='保留点 (最高 40%)', rasterized=True)
        m_all = np.array([r[cc == k, i].mean() for k in ucyc])
        m_keep = np.array([r[(cc == k) & keep, i].mean() if ((cc == k) & keep).any()
                           else r[cc == k, i].mean() for k in ucyc])
        ax.plot(ucyc, m_all, color='gray', ls='--', lw=1.6, label='循环平均 · 过滤前(200点)')
        ax.plot(ucyc, m_keep, color='#028090', lw=2.2, label='循环平均 · 过滤后(约80点)')
        if tru is not None:
            ax.axvline(tru, color='k', ls=':', lw=1.3, label=f'真实退化起点 (循环 {tru})')
        ax.set_title(f'{split} unit {u} — {s}   (drop {DROP:.0%}, seed 0)', fontsize=12)
        ax.set_xlabel('循环 (飞行次数)')
        ax.set_ylabel('残差 r = X - Xhat')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left', framealpha=0.9)
        fig.tight_layout()
        out = os.path.join(HERE, f'filter_drop60_{split}{u}_{s}.png')
        fig.savefig(out, dpi=150); plt.close(fig)
        print('saved', out)
