"""
exp_v2_resid.py — v2 chain, step 2: residual data + detcov validity by the
COLLEAGUE'S OWN METHOD (with exactly one change: the quadratic term uses
L22^-1 (solve) instead of L22 — 只改了 A 里面 L22 取逆).

Data: v2 baseline model (5 sensors, RBF rank1, cycle<3, 1000 rows/(unit,cycle),
      rebuilt from the stored training indices) -> for every dev unit and every
      cycle, 200 sampled rows (rng unit*7+seed, as always) -> per-point
      residual, detcov, corrected LL.  Cached to v2_stats_s{seed}.npz.

Detection, colleague style (NO baseline window, NO clipping):
      cycle curve = MEAN of the per-point LL over the cycle
      filter      = keep points with detcov > V   (their np.where(detcov>3.60);
                    V for our scale swept over absolute values set at pooled dev
                    percentiles {5,10,15,25} — reported as absolute numbers)
      change point = mean-drop statistic over the whole curve (their
                    mean_drop_changepoint, k from 5, upper bound extended to the
                    full range because DS03 onsets reach cycle 38)
Score: state-identification RMSE = onset RMSE vs the hs transition,
       9 dev units x 3 seeds, filter off vs each V, paired.
Also draws the colleague-style figure (raw vs filtered per-cycle mean residual,
5 sensors) for one unit per seed-0 model.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 8)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
os.environ['SENSORS'] = 'T30,T48,T50,Nc,Wf'
os.environ['KERNEL'] = 'rbf'
os.environ['RANK'] = '1'
from exp_traindata_detcov import fit_gp, SIDX, MCOND, DT
from demo_cond2 import conditional_stats2

SENS = ['T30', 'T48', 'T50', 'Nc', 'Wf']
NPER = 200
SEEDS = [0, 1, 2]
PCTS = [5, 10, 15, 25]          # pooled-dev percentiles defining absolute V's
FIG_UNIT = 7
RES = os.path.join(HERE, 'v2_resid_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


@torch.no_grad()
def point_stats(gp, Wq, Xq, chunk=4096):
    preds, dcs, lls = [], [], []
    for i in range(0, len(Wq), chunk):
        teX = torch.tensor(gp['xs'].transform(Wq[i:i + chunk]), dtype=DT)
        teY = torch.tensor(gp['ys'].transform(Xq[i:i + chunk]), dtype=DT)
        p, dc, _, llf, _ = conditional_stats2(gp['model'], gp['lik'], gp['tX'],
                                              gp['tY'], gp['struct'], teX,
                                              m=MCOND, test_y=teY)
        preds.append(p * gp['ys'].scale_ + gp['ys'].mean_)
        dcs.append(dc); lls.append(llf)
    return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


def meandrop_colleague(x, kmin=5):
    """colleague's mean_drop_changepoint, upper bound extended to full range."""
    n = len(x); s = np.std(x, ddof=1) + 1e-12; cs = np.cumsum(x)
    best = (None, -np.inf)
    for k in range(kmin, n - 1):
        m1 = cs[k - 1] / k; m2 = (cs[-1] - cs[k - 1]) / (n - k)
        t = np.sqrt(k * (n - k) / n) * (m1 - m2) / s
        if t > best[1]:
            best = (k, t)
    return best[0]


def build_stats(sd):
    path = os.path.join(HERE, f'v2_stats_s{sd}.npz')
    if os.path.exists(path):
        return np.load(path)
    cache = L.load_cache()
    W, X, A = cache['W_dev'], cache['X_s_dev'][:, SIDX], cache['A_dev']
    unit = A[:, 0].astype(int); cyc = A[:, 1].astype(int); hs = A[:, 3]
    z = np.load(os.path.join(HERE, f'v2_base_train_s{sd}.npz'))
    t0 = time.time()
    gp = fit_gp(W[z['train_idx']], X[z['train_idx']])
    log(f'seed{sd}: model rebuilt from stored indices ({time.time()-t0:.0f}s)')
    out = {}
    for u in np.unique(unit):
        rows = np.where(unit == u)[0]
        cyc_u = cyc[rows]; hs_u = hs[rows]
        rng_u = np.random.default_rng(int(u) * 7 + sd)
        ucyc = np.unique(cyc_u)
        idx, cc = [], []
        for c in ucyc:
            r = rows[cyc_u == c]
            if len(r) > NPER:
                r = rng_u.choice(r, NPER, replace=False)
            idx.append(r); cc.append(np.full(len(r), c))
        idx = np.concatenate(idx); cc = np.concatenate(cc)
        pred, dcv, ll = point_stats(gp, W[idx], X[idx])
        hs_by = np.array([hs_u[cyc_u == c].mean() for c in ucyc])
        below = np.where(hs_by < 0.5)[0]
        onset = int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)
        out[f'u{u}_cc'] = cc.astype(np.int32)
        out[f'u{u}_resid'] = (X[idx] - pred).astype(np.float32)
        out[f'u{u}_dc'] = dcv.astype(np.float32)
        out[f'u{u}_ll'] = ll.astype(np.float32)
        out[f'u{u}_onset'] = np.array([onset])
    np.savez(path, **out)
    log(f'seed{sd}: point stats cached ({time.time()-t0:.0f}s total)')
    return np.load(path)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log('V2 STEP 2 — residuals (200/cycle) + detcov validity, colleague method '
        '(mean LL + full-range mean-drop; only the L22 inverse fixed)')

    Z = {sd: build_stats(sd) for sd in SEEDS}
    units = {sd: sorted({int(k[1:].split('_')[0]) for k in Z[sd].files
                         if k.endswith('_cc')}) for sd in SEEDS}

    # ---- absolute V grid from the pooled dev detcov ----
    Vs = {}
    for sd in SEEDS:
        dcp = np.concatenate([Z[sd][f'u{u}_dc'] for u in units[sd]])
        Vs[sd] = {p: float(np.percentile(dcp, p)) for p in PCTS}
        log(f'seed{sd}: detcov pct[5,50,95]='
            f'{np.percentile(dcp, [5, 50, 95]).round(3)}  '
            f'V grid={[round(Vs[sd][p],3) for p in PCTS]}')

    # ---- detection: filter off vs each V ----
    def errors(filter_pct):
        e = []
        for sd in SEEDS:
            for u in units[sd]:
                cc = Z[sd][f'u{u}_cc']; ll = Z[sd][f'u{u}_ll']
                dc = Z[sd][f'u{u}_dc']; onset = int(Z[sd][f'u{u}_onset'][0])
                ucyc = np.unique(cc)
                cur = np.empty(len(ucyc))
                for i, c in enumerate(ucyc):
                    b = cc == c
                    if filter_pct is None:
                        m = b
                    else:
                        m = b & (dc > Vs[sd][filter_pct])
                        if not m.any():
                            m = b
                    cur[i] = ll[m].mean()
                k = meandrop_colleague(cur)
                det = int(ucyc[k]) if (k is not None and k < len(ucyc)) else int(ucyc[-1] + 1)
                e.append(det - onset)
        return np.array(e, float)

    log('')
    log('state-identification RMSE (cycles; 9 dev units x 3 seeds = 27 cases)')
    e_off = errors(None)
    log(f'  {"no filter":>12}: RMSE={np.sqrt((e_off**2).mean()):6.2f}  '
        f'mean delay={e_off.mean():+6.2f}  |d|<=3: {100*(np.abs(e_off)<=3).mean():3.0f}%')
    best = None
    for p in PCTS:
        e = errors(p)
        rm = np.sqrt((e ** 2).mean())
        pw = st.wilcoxon(np.abs(e), np.abs(e_off), zero_method='zsplit').pvalue
        log(f'  {"V@p"+str(p):>12}: RMSE={rm:6.2f}  mean delay={e.mean():+6.2f}  '
            f'|d|<=3: {100*(np.abs(e)<=3).mean():3.0f}%   vs off: '
            f'better {int((np.abs(e)<np.abs(e_off)).sum())} worse '
            f'{int((np.abs(e)>np.abs(e_off)).sum())}  Wilcoxon p={pw:.3f}')
        if best is None or rm < best[1]:
            best = (p, rm, e)
    log('')
    bp = best[0]
    log(f'best filter: V@p{bp} (abs {[round(Vs[sd][bp],2) for sd in SEEDS]})  '
        f'RMSE {best[1]:.2f}  vs no-filter {np.sqrt((e_off**2).mean()):.2f}')

    # ---- colleague-style figure: raw vs filtered mean residual, seed0 ----
    sd = 0; u = FIG_UNIT
    cc = Z[sd][f'u{u}_cc']; rs = Z[sd][f'u{u}_resid']; dc = Z[sd][f'u{u}_dc']
    ucyc = np.unique(cc); V = Vs[sd][bp]
    raw = np.empty((len(ucyc), 5)); fil = np.empty((len(ucyc), 5))
    for i, c in enumerate(ucyc):
        b = cc == c
        raw[i] = rs[b].mean(0)
        m = b & (dc > V)
        fil[i] = rs[m if m.any() else b].mean(0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for j, s in enumerate(SENS):
        ax = axes.flat[j]
        ax.plot(raw[:, j], '.', color='tab:blue', label='Raw')
        ax.plot(fil[:, j], '-', color='tab:orange', lw=1.5, label='Filter')
        ax.set_title(s)
    axes.flat[5].axis('off')
    axes.flat[5].legend(*axes.flat[0].get_legend_handles_labels(),
                        loc='center', fontsize=14, frameon=True)
    fig.suptitle(f'unit {u}, seed 0 — per-cycle mean residual, '
                 f'filter detcov > {V:.2f} (p{bp})')
    fig.tight_layout()
    p_ = os.path.join(HERE, f'fig_v2_meanresid_u{u}.png')
    fig.savefig(p_, dpi=140); plt.close(fig)
    log(f'figure: {p_}')
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
