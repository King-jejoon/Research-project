"""make_cs_data.py — extract every number the Case-study section needs
into case_study_data.json and draw the new figures (Fig 6, Fig 9, Fig 4).
DS03 from the recorded v5c/v5b files; transfer subsets from {ds}v6_*.npz
when present.  Score = NASA asymmetric sum per seed, mean over 3 seeds."""
import os, json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

E = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'experiments'))
OUT = os.path.dirname(os.path.abspath(__file__))
FR = [0.2, 0.4, 0.6, 0.8]
SEEDS = 3


def nasa(P, T):
    d = np.asarray(P, float) - np.asarray(T, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def levels(P, T, F):
    """per-level (RMSE, score) + overall; F is fracs-major within each seed."""
    n = len(P) // SEEDS
    out = {}
    for f in FR + ['all']:
        m = np.ones(len(P), bool) if f == 'all' else np.isclose(F, f)
        rm = float(np.sqrt(((P[m] - T[m]) ** 2).mean()))
        sc = float(np.mean([nasa(P[i*n:(i+1)*n][m[i*n:(i+1)*n]],
                                 T[i*n:(i+1)*n][m[i*n:(i+1)*n]])
                            for i in range(SEEDS)]))
        seeds_rm = [float(np.sqrt(((P[i*n:(i+1)*n][m[i*n:(i+1)*n]] -
                                    T[i*n:(i+1)*n][m[i*n:(i+1)*n]]) ** 2).mean()))
                    for i in range(SEEDS)]
        # report convention: per-level RMSE pooled; overall = mean of per-seed RMSE
        out[str(f)] = dict(rmse=(float(np.mean(seeds_rm)) if f == 'all' else rm),
                           score=sc, rmse_sd=float(np.std(seeds_rm)))
    return out


def per_unit_level_rmse(P, T, F, nunits):
    """per-unit RMSE per level over seeds (for boxplots); order within a
    seed block = fracs-major, units-minor."""
    n = len(P) // SEEDS
    res = {str(f): [] for f in FR + ['all']}
    for ui in range(nunits):
        for fi, f in enumerate(FR):
            idx = [i*n + fi*nunits + ui for i in range(SEEDS)]
            res[str(f)].append(float(np.sqrt(((P[idx] - T[idx]) ** 2).mean())))
        idx = [i*n + fi*nunits + ui for i in range(SEEDS) for fi in range(4)]
        res['all'].append(float(np.sqrt(((P[idx] - T[idx]) ** 2).mean())))
    return res


data = {}
# ---------------- DS03 ----------------
d3 = {}
zp = np.load(f'{E}/v5c_test.npz'); zb = np.load(f'{E}/v5b_test_batch.npz')
zt = np.load(f'{E}/v5c_timing.npz'); za = np.load(f'{E}/v5c_test_ablate.npz')
d3['bench'] = {
    'Proposed': dict(**levels(zp['P'], zp['T'], zp['F']),
                     train=float(zt['proposed_train']), infer=float(zt['proposed_infer'])),
    'MOGP': dict(**levels(zb['ungated_P'], zb['ungated_T'], zb['ungated_F']),
                 train=float(zt['mogp_train']), infer=float(zt['mogp_infer'])),
}
for k, lab in [('llke', 'LLKE'), ('bspline', 'B-spline'), ('lr', 'LR'), ('cabn', 'CaBN')]:
    d3['bench'][lab] = dict(**levels(zb[f'{k}_P'], zb[f'{k}_T'], zb[f'{k}_F']),
                            train=float(zt[f'{k}_train']), infer=float(zt[f'{k}_infer']))
# ablation: main modules
d3['abl_main'] = {
    'Proposed': levels(zp['P'], zp['T'], zp['F']),
    'Without the state division': levels(zb['ungated_P'], zb['ungated_T'], zb['ungated_F']),
}
raw_p = f'{E}/v5c_raw_resummary.npz'
if os.path.exists(raw_p):
    zr = np.load(raw_p)
    d3['abl_main']['Without the residual extraction'] = levels(zr['P'], zr['T'], zr['F'])
# ablation: HI components (v5c_test_ablate has P/T, F = chain F)
F3 = zp['F']
d3['abl_hi'] = {
    'Proposed': levels(za['base_P'], za['base_T'], F3),
    'Without the initial-level term': levels(za['no_l0_P'], za['no_l0_T'], F3),
    'Without the end-level term': levels(za['no_end_P'], za['no_end_T'], F3),
    'Without the tail-flatness term': levels(za['no_flat_P'], za['no_flat_T'], F3),
}
# lambda grid (stage-2 tables) -> Table 2 + Fig 9
zl = np.load(f'{E}/v5c_lambda_grid.npz')
L1S = [2.0, 4.0, 6.0, 8.0, 12.0, 16.0]; L2S = [0.25, 0.5, 1.0, 2.0, 4.0]
grid = {}
for l1 in L1S:
    for l2 in L2S:
        k = f'{l1}_{l2}'
        grid[k] = dict(mon=float(zl[f'{k}_mon']), curv=float(zl[f'{k}_curv']),
                       rng=float(zl[f'{k}_rng']), rul=float(zl[f'{k}_rul']))
d3['lambda_grid'] = grid
# onset per-unit RMSE test (Fig 4) + dev numbers
e = zp['e_on_off'].reshape(SEEDS, -1)          # seed-major, 6 units
d3['onset_unit_rmse_test'] = [float(np.sqrt((e[:, j] ** 2).mean())) for j in range(e.shape[1])]
d3['onset'] = dict(dev_cond=5.21, dev_ung=6.78, test_cond=8.54, test_ung=9.01,
                   dev_p='0.003')
d3['rul_unit_level_test'] = per_unit_level_rmse(zp['P'], zp['T'], zp['F'], 6)
d3['hparams'] = dict(sensors='T30, T48, T50, Nc, Wf', kernel='RBF, rank 1',
                     m=18, window='30 flight-hours', clip='10 sigma',
                     V='V_sparse = 19.0 (nbase < 10), V_dense = 20.2',
                     lam='lambda1 = 12, lambda2 = 0.25', beta='30/30/30',
                     budget='1,000 rows per unit-cycle (27k); stage 2: 3,000 rows per unit (27k)',
                     recipe='lambda0 = 1, thr = 0.20, end_target = 1.03, flat w = 800, m = 0.004, med3')
data['ds03'] = d3

# ---------------- transfer subsets ----------------
for ds in ['ds08a', 'ds07']:
    p = f'{E}/{ds}v6_test.npz'
    if not os.path.exists(p):
        continue
    z = np.load(p)
    dd = {}
    nunits = len(set(z['on_ids'][:, 1].tolist()))
    dd['bench'] = {}
    for k, lab in [('hr', 'Proposed'), ('c3', 'MOGP'), ('llke', 'LLKE'),
                   ('bspline', 'B-spline'), ('lr', 'LR'), ('cabn', 'CaBN')]:
        dd['bench'][lab] = levels(z[f'{k}_P'], z[f'{k}_T'], z[f'{k}_F'])
    dd['abl_main'] = {'Proposed': dd['bench']['Proposed'],
                      'Without the state division': dd['bench']['MOGP']}
    er = z['e_on_rule'].reshape(SEEDS, -1); eu = z['e_on_ug'].reshape(SEEDS, -1)
    dd['onset_unit_rmse_test'] = [float(np.sqrt((er[:, j] ** 2).mean())) for j in range(er.shape[1])]
    zr_ = np.load(f'{E}/{ds}v6_detsel.npz')
    dd['onset'] = dict(dev_cond=float(zr_['r_rule'][0]), dev_ung=float(zr_['r_ug'][0]),
                       test_cond=float(np.sqrt((er ** 2).mean())),
                       test_ung=float(np.sqrt((eu ** 2).mean())))
    dd['rul_unit_level_test'] = per_unit_level_rmse(z['hr_P'], z['hr_T'], z['hr_F'], nunits)
    dd['rule'] = dict(NB=int(zr_['NB'][0]), V_sparse=float(zr_['V_sparse'][0]),
                      V_dense=float(zr_['V_dense'][0]), wval=float(zr_['wval'][0]),
                      wmode=str(zr_['wmode'][0]), clip=float(zr_['clip'][0]))
    zh1 = np.load(f'{E}/{ds}v6_hi1.npz'); zh2 = np.load(f'{E}/{ds}v6_stage2.npz')
    dd['recipe1'] = [float(zh1['l1'][0]), float(zh1['l2'][0])]
    dd['recipe2'] = [float(zh2['l1'][0]), float(zh2['l2'][0])]
    dd['dev_hi1'] = [float(zh1['dev_rm'].mean()), float(zh1['dev_rm'].std())]
    dd['dev_hi2'] = [float(zh2['rms'].mean()), float(zh2['rms'].std())]
    dd['nunits_test'] = nunits
    import re
    txt = open(f'{E}/{ds}v6_stage2_results.txt').read()
    m1 = re.search(r'healthy floor RMS : c3 ([\d.]+)\s+hr ([\d.]+)\s+ratio ([\d.]+).*?hr lower in (\d+)/(\d+)', txt)
    m2 = re.search(r'SNR signal/floor  : c3 ([\d.]+)\s+hr ([\d.]+).*?hr better in (\d+)/(\d+)', txt)
    dd['snr'] = dict(floor_ratio=float(m1.group(3)), floor_won=f'{m1.group(4)}/{m1.group(5)}',
                     snr_c3=float(m2.group(1)), snr_hr=float(m2.group(2)), snr_won=f'{m2.group(3)}/{m2.group(4)}')
    data[ds] = dd

json.dump(data, open(f'{OUT}/case_study_data.json', 'w'), indent=1)

# ---------------- figures ----------------
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})
LAB = {'ds03': 'DS03', 'ds01': 'DS01', 'ds02': 'DS02', 'ds08a': 'DS08a', 'ds07': 'DS07'}
# Fig 4: onset RMSE of in-service units per sub-dataset
keys = [k for k in ['ds03', 'ds08a', 'ds07'] if k in data]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.boxplot([data[k]['onset_unit_rmse_test'] for k in keys], labels=[LAB[k] for k in keys],
           whis=(0, 100), widths=0.5)
for i, k in enumerate(keys):
    ax.scatter(np.full(len(data[k]['onset_unit_rmse_test']), i + 1),
               data[k]['onset_unit_rmse_test'], s=14, color='k', zorder=3)
ax.set_xlabel('Sub-dataset'); ax.set_ylabel('Onset RMSE (cycles)')
ax.grid(axis='y', alpha=0.3); fig.tight_layout()
fig.savefig(f'{OUT}/cs_fig4_onset_subsets.png', dpi=200); plt.close(fig)
# Fig 6: RUL RMSE over levels (DS03 test, per-unit boxes)
r = data['ds03']['rul_unit_level_test']
fig, ax = plt.subplots(figsize=(5.2, 3.4))
lv = ['0.2', '0.4', '0.6', '0.8', 'all']
ax.boxplot([r[l] for l in lv], labels=['20%', '40%', '60%', '80%', 'Average'],
           whis=(0, 100), widths=0.5)
for i, l in enumerate(lv):
    ax.scatter(np.full(len(r[l]), i + 1), r[l], s=14, color='k', zorder=3)
ax.set_xlabel('RUL level'); ax.set_ylabel('RUL RMSE (cycles)')
ax.grid(axis='y', alpha=0.3); fig.tight_layout()
fig.savefig(f'{OUT}/cs_fig6_rul_levels_ds03.png', dpi=200); plt.close(fig)
# Fig 9: lambda heatmap (dev LOO RUL, stage-2 tables)
M = np.array([[grid[f'{l1}_{l2}']['rul'] for l1 in L1S] for l2 in L2S])
fig, ax = plt.subplots(figsize=(5.2, 3.6))
im = ax.imshow(M, cmap='Blues_r', aspect='auto', origin='lower')
ax.set_xticks(range(len(L1S))); ax.set_xticklabels([str(int(v)) for v in L1S])
ax.set_yticks(range(len(L2S))); ax.set_yticklabels([str(v) for v in L2S])
ax.set_xlabel('lambda1'); ax.set_ylabel('lambda2')
for i in range(len(L2S)):
    for j in range(len(L1S)):
        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if M[i, j] < M.min() + 0.35 * (M.max() - M.min()) else 'black')
ax.add_patch(plt.Rectangle((L1S.index(12.0) - 0.5, L2S.index(0.25) - 0.5), 1, 1,
                           fill=False, lw=2, color='red'))
fig.colorbar(im, ax=ax, label='dev LOO RUL RMSE (cycles)'); fig.tight_layout()
fig.savefig(f'{OUT}/cs_fig9_lambda_heatmap.png', dpi=200); plt.close(fig)
print('datasets:', list(data), '-> case_study_data.json + figs')
