"""
exp_v3_final20.py — v3 chain finalized at the ABSOLUTE gate V* = 20.0
(user decision: grid minimum of the absolute-covariance sweep; 8.68 -> 7.16,
p=0.002, consistent across all 3 seeds and all 4 discordant cases).

Recomputes everything downstream of the gate from cached per-point stats
(no GP refit):
  D  lambda1 x lambda2 grid (properties + shape + LOO RUL) at V=20.0
     -> selection by the frozen rule (shape gate, then property composite)
     -> HI curves figure
  E  dev LOO truncation RUL for the selected combo (per-fraction)
  C  extended-low detector sensitivity grid at V=20.0 -> two figures
  B  gate-sweep figure on the absolute axis (V=19.4..20.3, star at 20.0)
  T  TEST re-verification at V=20.0 (third opening of DS03 test as a dataset,
     recorded): detection + truncation RUL, cached test point stats reused.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import (HI_CFG, BETA_C, FRACS, nasa, loo, eval_units,
                         train_model_tail, shape_ok)
from exp_v3_d_hi import build_tables, properties
from exp_v3_a2cmp import load as load_stats, SEEDS
from exp_resid_clean import meandrop

VSTAR = 20.0
L1S = [2.0, 4.0, 8.0, 16.0]
L2S = [0.5, 1.0, 2.0, 4.0]
OUT = __import__('exp_paths').PAPER
RES = os.path.join(HERE, 'v3_final20_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def trim25(v):
    s = np.sort(v, axis=0); n = len(v)
    return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean(0)


def detect(cur, dur, ucyc, wh=30.0, clip=10.0):
    w = max(5, int(np.searchsorted(np.cumsum(dur), wh) + 1))
    mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
    x = np.maximum(cur, mu0 - clip * sd0)
    k = meandrop(x)
    return int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'V3 FINAL CHAIN AT ABSOLUTE V* = {VSTAR}')

    # ---------------- D: lambda grid ----------------
    data = {sd: build_tables(sd, VSTAR) for sd in SEEDS}
    log('')
    log('D — lambda1 x lambda2 grid (gate V=20.0)')
    R = {}
    for l1 in L1S:
        for l2 in L2S:
            cfg = dict(HI_CFG); cfg.update(lambda1=l1, lambda2=l2, end_target=1.03)
            props, rmses, oks = [], [], []
            for sd in SEEDS:
                units, Zn, hrs = data[sd]
                np.random.seed(sd)
                him, _ = train_model_tail([Zn[u] for u in units], **cfg)
                HIs = {u: him.forward(Zn[u]).flatten() for u in units}
                ok, _ = shape_ok(HIs)
                P, T, F, bsel, _ = loo('cycle', units, HIs, hrs, BETA_C)
                props.append(properties(HIs))
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                oks.append(ok)
            pr = np.array(props).mean(0)
            R[(l1, l2)] = dict(mon=pr[0], curv=pr[1], rng=pr[2], shift=pr[3],
                               rul=float(np.mean(rmses)), rul_sd=float(np.std(rmses)),
                               shape=int(np.sum(oks)))
            log(f'  l1={l1:>4} l2={l2:>3}: Mon={pr[0]:8.2f} Curv={pr[1]:6.2f} '
                f'Range={pr[2]:5.3f} Shift={pr[3]:6.4f} '
                f'RUL={np.mean(rmses):.2f}±{np.std(rmses):.2f} shape {int(np.sum(oks))}/3')

    ok_keys = [k for k in R if R[k]['shape'] == 3]
    pool = ok_keys if ok_keys else list(R)
    arr = {p: np.array([R[k][p] for k in pool]) for p in ['mon', 'curv', 'rng', 'shift']}
    nrm = lambda v, sg: (sg * v - (sg * v).min()) / ((sg * v).max() - (sg * v).min() + 1e-12)
    comp = (nrm(arr['mon'], 1) + nrm(arr['curv'], 1) + nrm(arr['rng'], 1) + nrm(arr['shift'], -1)) / 4
    sel = pool[int(np.argmax(comp))]
    log(f'  shape-passing: {sorted(ok_keys)}')
    log(f'  SELECTED: lambda1={sel[0]}, lambda2={sel[1]} (composite {comp.max():.3f})  '
        f'RUL {R[sel]["rul"]:.2f}±{R[sel]["rul_sd"]:.2f} '
        f'(grid min RUL {min(r["rul"] for r in R.values()):.2f})')

    # HI curves figure
    cfg = dict(HI_CFG); cfg.update(lambda1=sel[0], lambda2=sel[1], end_target=1.03)
    units, Zn, hrs = data[0]
    np.random.seed(0)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in units:
        h = him.forward(Zn[u]).flatten()
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title(f'HI trajectories (dev, seed 0; lambda1={sel[0]:g}, lambda2={sel[1]:g}, V*=20.0)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_v3_hi_curves.png'), dpi=150)
    plt.close(fig)

    # ---------------- E: dev LOO for selected ----------------
    log('')
    log('E — dev LOO truncation RUL (selected lambda)')
    allP, allT, allF, rms = [], [], [], []
    betas = []
    for sd in SEEDS:
        units, Zn, hrs = data[sd]
        np.random.seed(sd)
        him, _ = train_model_tail([Zn[u] for u in units], **cfg)
        HIs = {u: him.forward(Zn[u]).flatten() for u in units}
        P, T, F, b, _ = loo('cycle', units, HIs, hrs, BETA_C)
        rms.append(float(np.sqrt(((P - T) ** 2).mean()))); betas.append(b)
        allP.append(P); allT.append(T); allF.append(F)
        log(f'  seed{sd}: RMSE={rms[-1]:.2f} NASA={nasa(P,T):.0f} beta={b}')
    P = np.concatenate(allP); T = np.concatenate(allT); F = np.concatenate(allF)
    log(f'  dev overall: {np.mean(rms):.2f} ± {np.std(rms):.2f}   '
        f'per-frac: ' + '  '.join(
            f'{int(f*100)}%={np.sqrt(((P[np.isclose(F,f)]-T[np.isclose(F,f)])**2).mean()):.2f}'
            for f in FRACS))

    # ---------------- C: sensitivity at V=20.0 ----------------
    log('')
    log('C — extended-low sensitivity grid (gate V=20.0)')
    D5 = {sd: load_stats(5, sd) for sd in SEEDS}
    curves = {}
    for sd in SEEDS:
        for u, d in D5[sd].items():
            cur = np.empty(len(d['ucyc']))
            for i, c in enumerate(d['ucyc']):
                b = d['cc'] == c
                m = b & (d['dc'] >= VSTAR)
                if not m.any():
                    m = b
                cur[i] = np.percentile(d['ll'][m], 75)
            curves[(sd, u)] = (cur, d['dur'], d['ucyc'], d['onset'])
    CYC = [3, 4, 5, 6, 8, 12, 20]; FH = [10, 15, 20, 25, 30, 50]
    CLIPS = [10, 12, 14, 16]
    SR = {}
    for mode, wins in [('cyc', CYC), ('fh', FH)]:
        for wdw in wins:
            for cl in CLIPS:
                e = []
                for cur, dur, ucyc, onset in curves.values():
                    if mode == 'cyc':
                        w = max(3, min(int(wdw), len(cur) - 5))
                        mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
                    else:
                        w = max(3, int(np.searchsorted(np.cumsum(dur), float(wdw)) + 1))
                        mu0, sd0 = cur[:w].mean(), cur[:w].std(ddof=1) + 1e-8
                    k = meandrop(np.maximum(cur, mu0 - cl * sd0))
                    det = int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)
                    e.append(det - onset)
                e = np.array(e, float)
                SR[(mode, wdw, cl)] = float(np.sqrt((e ** 2).mean()))
    log('  clip10 row: ' + '  '.join(f'{w}cy={SR[("cyc",w,10)]:.2f}' for w in CYC) +
        ' | ' + '  '.join(f'{w}h={SR[("fh",w,10)]:.2f}' for w in FH))
    MARK = ['o', 's', '^', 'D', 'v', 'P', 'X']
    for mode, wins, unit, title, fname in [
            ('cyc', CYC, 'cycles', 'Cycle-count baseline windows', 'fig_v3_sens_cycle_ext.png'),
            ('fh', FH, 'flight-h', 'Flight-hour baseline windows', 'fig_v3_sens_flighth_ext.png')]:
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        for i, w in enumerate(wins):
            ax.plot(CLIPS, [SR[(mode, w, cl)] for cl in CLIPS], marker=MARK[i % 7],
                    ls='--', ms=6, lw=1.4, label=f'{w} {unit}')
        ax.set_xlabel('Clip strength  c'); ax.set_ylabel('Onset RMSE [cycles]')
        ax.set_title(title); ax.set_xticks(CLIPS); ax.grid(True, ls=':', alpha=0.5)
        ax.legend(title='Baseline window', frameon=False, fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=150); plt.close(fig)

    # ---------------- B: gate-sweep figure, absolute axis ----------------
    from exp_v3_b_detcov import errors as gate_errors
    GRID = [19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20.0, 20.1, 20.2, 20.3]
    e_off = gate_errors(D5, None); r_off = np.sqrt((e_off ** 2).mean())
    rms_g, pvals = [], []
    for V in GRID:
        e = gate_errors(D5, {sd: V for sd in SEEDS})
        rms_g.append(np.sqrt((e ** 2).mean()))
        pvals.append(st.wilcoxon(np.abs(e), np.abs(e_off), zero_method='zsplit').pvalue)
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    ax.axhline(r_off, color='gray', lw=1.2, label=f'no gate ({r_off:.2f})')
    ax.plot(GRID, rms_g, marker='o', ls='--', color='tab:blue')
    for x, y, p_ in zip(GRID, rms_g, pvals):
        if p_ < 0.05 and y < r_off:
            ax.plot(x, y, marker='*', ms=13, color='tab:red')
    ax.plot([VSTAR], [rms_g[GRID.index(VSTAR)]], marker='o', ms=11, mfc='none',
            mec='k', mew=1.6)
    ax.set_xlabel('absolute detcov threshold  V  (keep detcov >= V)')
    ax.set_ylabel('Onset RMSE [cycles]')
    ax.set_title('Inspection-gate sweep, absolute axis (* p<0.05; circle = V* = 20.0)')
    ax.grid(True, ls=':', alpha=0.5); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_v3_gate_sweep.png'), dpi=150)
    plt.close(fig)
    log('B — gate figure regenerated (absolute axis)')

    # ---------------- T: test re-verification ----------------
    log('')
    log('T — TEST re-verification at V*=20.0 (3rd opening, recorded)')
    H = {sd: np.load(os.path.join(HERE, f'rul_input_s{sd}.npz')) for sd in SEEDS}
    det_err = []
    rul_R, rul_N = [], []
    aP, aT, aF = [], [], []
    for sd in SEEDS:
        Zd = np.load(os.path.join(HERE, f'v2_stats_s{sd}.npz'))
        d_units = sorted({int(k[1:].split('_')[0]) for k in Zd.files if k.endswith('_cc')})
        dev_raw, dev_hrs = {}, {}
        for u in d_units:
            cc = Zd[f'u{u}_cc']; rs = Zd[f'u{u}_resid']; dc = Zd[f'u{u}_dc']
            ur = H[sd][f'dev_{u}_ucyc']; durs = H[sd][f'dev_{u}_hours']
            pos = {int(c): i for i, c in enumerate(ur)}
            ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
            Tm = np.empty((len(ucyc), rs.shape[1]))
            for i, c in enumerate(ucyc):
                b = cc == c
                m = b & (dc >= VSTAR)
                Tm[i] = trim25(rs[m if m.any() else b])
            dev_raw[u] = Tm
            dev_hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc]))
        Zt = np.load(os.path.join(HERE, f'v3_test_stats_s{sd}.npz'))
        t_units = sorted({int(k[1:].split('_')[0]) for k in Zt.files if k.endswith('_cc')})
        t_raw, t_hrs, t_llq, t_uc, t_on, t_du = {}, {}, {}, {}, {}, {}
        for u in t_units:
            cc = Zt[f'u{u}_cc']; rs = Zt[f'u{u}_resid']; dc = Zt[f'u{u}_dc']
            ll = Zt[f'u{u}_ll']; ucyc = Zt[f'u{u}_ucyc']; d_ = Zt[f'u{u}_hours'].astype(float)
            Tm = np.empty((len(ucyc), rs.shape[1])); q = np.empty(len(ucyc))
            for i, c in enumerate(ucyc):
                b = cc == c
                m = b & (dc >= VSTAR)
                if not m.any():
                    m = b
                Tm[i] = trim25(rs[m]); q[i] = np.percentile(ll[m], 75)
            t_raw[u] = Tm; t_hrs[u] = np.cumsum(d_); t_llq[u] = q
            t_uc[u] = ucyc; t_on[u] = int(Zt[f'u{u}_onset'][0]); t_du[u] = d_
        row = []
        for u in t_units:
            det = detect(t_llq[u], t_du[u], t_uc[u])
            row.append(f'u{u}:t{t_on[u]}/d{det}({det-t_on[u]:+d})')
            det_err.append(det - t_on[u])
        log(f'  seed{sd} onset: ' + '  '.join(row))
        allr = np.concatenate([dev_raw[u] for u in d_units])
        mu, sg = allr.mean(0), allr.std(0) + 1e-8
        Zn_d = {u: (dev_raw[u] - mu) / sg for u in d_units}
        Zn_t = {u: (t_raw[u] - mu) / sg for u in t_units}
        np.random.seed(sd)
        him, _ = train_model_tail([Zn_d[u] for u in d_units], **cfg)
        HI_d = {u: him.forward(Zn_d[u]).flatten() for u in d_units}
        HI_t = {u: him.forward(Zn_t[u]).flatten() for u in t_units}
        _, _, _, beta, _ = loo('cycle', d_units, HI_d, dev_hrs, BETA_C)
        cd = {u: np.arange(len(HI_d[u])) / 500.0 for u in d_units}
        ct = {u: np.arange(len(HI_t[u])) / 500.0 for u in t_units}
        P_, T_, F_, _ = eval_units(beta, cd, HI_d, ct, HI_t, FRACS)
        rul_R.append(float(np.sqrt(((P_ - T_) ** 2).mean())))
        rul_N.append(nasa(P_, T_))
        aP.append(P_); aT.append(T_); aF.append(F_)
        log(f'  seed{sd} RUL: beta={beta}  RMSE={rul_R[-1]:.2f}  NASA={rul_N[-1]:.1f}')
    e = np.array(det_err, float)
    log(f'  TEST onset: RMSE={np.sqrt((e**2).mean()):.2f}  delay={e.mean():+.2f}  '
        f'|d|<=3: {100*(np.abs(e)<=3).mean():.0f}%')
    P_ = np.concatenate(aP); T_ = np.concatenate(aT); F_ = np.concatenate(aF)
    log(f'  TEST RUL: {np.mean(rul_R):.2f} ± {np.std(rul_R):.2f}  '
        f'NASA {np.mean(rul_N):.1f} ± {np.std(rul_N):.1f}  per-frac: ' + '  '.join(
            f'{int(f*100)}%={np.sqrt(((P_[np.isclose(F_,f)]-T_[np.isclose(F_,f)])**2).mean()):.2f}'
            for f in FRACS))
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
