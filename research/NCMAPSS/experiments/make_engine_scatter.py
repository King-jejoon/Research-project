"""
make_engine_scatter.py  —  reference-style scatter thumbnails (rounded frame,
dots only, NO text) for dev engine #1 and test engine #1.

Per engine, into <out>/<engine>/:
  raw/        each sensor (14) and each operating condition W (4)  -- single colour
  partition/  same scatters, coloured by the UNSUPERVISED monitor's normal/abnormal
              split (two dot colours only)

Run:
  /opt/anaconda3/envs/pt_prac/bin/python3 make_engine_scatter.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from vecchia_gp import VecchiaGP

OUT = os.path.join(__import__('exp_paths').PAPER, 'engine_scatter')
HI_DIR = '/Users/a1/Desktop/Research-project/research/HI_RUL_prediction '
SENSORS = L.OUTPUT_NAMES
WNAMES = ['alt', 'Mach', 'TRA', 'T2']

BLUE = '#5b8bbf'      # raw / normal
RED = '#d33f34'       # abnormal

WARMUP = 10
K = 5.0
LAM = 0.2


def save_scatter(x, y, colors, path, single=True):
    fig = plt.figure(figsize=(2.7, 1.5))          # 1.8 : 1
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    x = np.asarray(x, float); y = np.asarray(y, float)
    xn = (x - x.min()) / (x.max() - x.min() + 1e-9)
    yn = (y - y.min()) / (y.max() - y.min() + 1e-9)
    px = 0.04 + 0.92 * xn
    py = 0.10 + 0.80 * yn
    ax.scatter(px, py, s=10, c=(BLUE if single else colors), alpha=0.62, linewidths=0)
    fig.savefig(path, dpi=200, transparent=True)   # no border, transparent
    plt.close(fig)


def per_cycle_groups(rows, cyc, n_per, seed):
    """Identical to exp_hs_monitor: sample up to n_per rows per cycle."""
    rng = np.random.default_rng(seed)
    ucyc = np.unique(cyc)
    sel = []
    for cc in ucyc:
        r = rows[cyc == cc]
        if len(r) > n_per:
            r = rng.choice(r, size=n_per, replace=False)
        sel.append(r)
    return ucyc, sel


def compute_Tc(vg, mu_r, sd_r, W, X, rows, cyc, seed, n_per=30):
    """Per-cycle mean T^2 statistic (the EWMA-charted quantity)."""
    ucyc, sel = per_cycle_groups(rows, cyc, n_per, seed)
    allidx = np.concatenate(sel)
    mean_p, _ = vg.predict(W[allidx])
    z = (X[allidx] - mean_p - mu_r) / sd_r
    t2 = (z ** 2).sum(1)
    off = 0; Tc = []
    for r in sel:
        n = len(r); Tc.append(t2[off:off + n].mean()); off += n
    return ucyc, np.array(Tc)


def detect(ucyc, Tc, mu0, sd0):
    UCL = mu0 + K * sd0 * np.sqrt(LAM / (2 - LAM))
    e = mu0; E = np.empty_like(Tc)
    for i, t in enumerate(Tc):
        e = LAM * t + (1 - LAM) * e; E[i] = e
    cross = np.where(E > UCL)[0]
    return int(ucyc[cross[0]]) if len(cross) else int(ucyc[-1] + 1)


def main():
    c = L.load_cache()
    Wd, Xd, Ad = c['W_dev'], c['X_s_dev'], c['A_dev']
    Wt, Xt, At = c['W_test'], c['X_s_test'], c['A_test']
    dev_normal_idx = c['dev_normal_idx']

    # pipeline baseline EXACTLY as in the code: gp_residuals.MultitaskGPModel
    # trained on dev_normal_idx (180 pts).  Residuals use THIS model.
    print('training pipeline GP baseline (MultitaskGP on dev_normal_idx) ...')
    gpm, gpl, gpsc = L.train_gp(Wd, Xd, dev_normal_idx, iters=100, seed=0)

    # ---- per engine: fit its OWN normal model on its run-in (self-calibrated) ----
    engines = [('dev1', Wd, Xd, Ad, 1), ('test1', Wt, Xt, At, int(At[:, 0].min()))]
    for tag, W, X, A, unit in engines:
        rows = np.where(A[:, 0].astype(int) == unit)[0]
        cyc = A[rows, 1].astype(int)
        # seed = this engine's first WARMUP cycles (assumed healthy)
        rng = np.random.default_rng(int(unit) * 13)
        seed = []
        for cc in range(1, WARMUP + 1):
            r = rows[cyc == cc]
            if len(r):
                seed.append(rng.choice(r, size=min(120, len(r)), replace=False))
        seed = np.concatenate(seed)
        vg = VecchiaGP(m=15).fit(W[seed], X[seed], iters=150, verbose=False)
        mean_s, _ = vg.predict(W[seed]); res_s = X[seed] - mean_s
        mu_r, sd_r = res_s.mean(0), res_s.std(0) + 1e-8
        ucyc, Tc = compute_Tc(vg, mu_r, sd_r, W, X, rows, cyc, seed=int(unit) * 91 + 7)
        ic_e = Tc[ucyc <= WARMUP]
        mu0_e, sd0_e = float(ic_e.mean()), float(ic_e.std() + 1e-8)
        trans = detect(ucyc, Tc, mu0_e, sd0_e)
        # ground-truth onset (sanity only)
        hs_on = '-'
        if A.shape[1] > 3:
            hsc = np.array([A[rows[cyc == cc], 3].mean() for cc in ucyc])
            bel = np.where(hsc < 0.5)[0]
            hs_on = int(ucyc[bel[0]]) if len(bel) else '-'
        print(f'{tag}: unit={unit}  cycles={cyc.min()}..{cyc.max()}  '
              f'pred_transition={trans}  true_onset(hs)={hs_on}')

        _, sgroups = per_cycle_groups(rows, cyc, 32, seed=2025)
        sel = np.concatenate(sgroups)
        scyc = A[sel, 1].astype(int)
        order = np.argsort(scyc); sel = sel[order]; scyc = scyc[order]
        jit = np.random.default_rng(0).uniform(-0.45, 0.45, len(sel))
        xx = scyc + jit
        labels_normal = scyc < trans
        part_colors = np.where(labels_normal, BLUE, RED)

        d_raw = os.path.join(OUT, tag, 'raw')
        d_part = os.path.join(OUT, tag, 'partition')
        os.makedirs(d_raw, exist_ok=True)
        os.makedirs(d_part, exist_ok=True)

        def norm(v):
            v = v.astype(float)
            return (v - v.min()) / (v.max() - v.min() + 1e-9)

        for j, name in enumerate(SENSORS):
            y = norm(X[sel, j])
            save_scatter(xx, y, None, os.path.join(d_raw, f'sensor_{j+1:02d}_{name}.png'), single=True)
            save_scatter(xx, y, part_colors, os.path.join(d_part, f'sensor_{j+1:02d}_{name}.png'), single=False)
        for k, name in enumerate(WNAMES):
            y = norm(W[sel, k])
            save_scatter(xx, y, None, os.path.join(d_raw, f'w_{k+1}_{name}.png'), single=True)
            save_scatter(xx, y, part_colors, os.path.join(d_part, f'w_{k+1}_{name}.png'), single=False)

        # ---- residual r = X - y_hat (Stage 3) using the PIPELINE GP, aggregated
        #      to per-cycle mean residual R_bar_n (exactly what Stage 4 consumes) ----
        d_res = os.path.join(OUT, tag, 'residual')
        d_resp = os.path.join(OUT, tag, 'residual_partition')
        os.makedirs(d_res, exist_ok=True); os.makedirs(d_resp, exist_ok=True)
        rucyc, rgroups = per_cycle_groups(rows, cyc, 60, seed=int(unit) * 5 + 1)
        rall = np.concatenate(rgroups)
        rmean, _ = L.gp_predict(gpm, gpl, gpsc, W, rall)
        rres = X[rall] - rmean
        off = 0; res_cyc = []
        for g in rgroups:
            n = len(g); res_cyc.append(rres[off:off + n].mean(0)); off += n
        res_cyc = np.array(res_cyc)                       # (n_cycle, 14)  = R_bar_n
        rcolors = np.where(rucyc < trans, BLUE, RED)
        for j, name in enumerate(SENSORS):
            yv = res_cyc[:, j]
            save_scatter(rucyc, yv, None, os.path.join(d_res, f'sensor_{j+1:02d}_{name}.png'), single=True)
            save_scatter(rucyc, yv, rcolors, os.path.join(d_resp, f'sensor_{j+1:02d}_{name}.png'), single=False)

        # ---- HI neural-network output (Stage 4) from saved CSV ----
        hidf = pd.read_csv(os.path.join(HI_DIR, 'dev_HI.csv' if tag == 'dev1' else 'test_HI.csv'))
        hu = 1 if tag == 'dev1' else int(hidf['Unit'].min())
        hi = hidf[hidf['Unit'] == hu]['HI'].values
        hcyc = np.arange(1, len(hi) + 1)
        hcolors = np.where(hcyc < trans, BLUE, RED)
        d_hi = os.path.join(OUT, tag, 'hi'); os.makedirs(d_hi, exist_ok=True)
        save_scatter(hcyc, hi, None, os.path.join(d_hi, 'HI.png'), single=True)
        save_scatter(hcyc, hi, hcolors, os.path.join(d_hi, 'HI_partition.png'), single=False)
        print(f'  saved raw(18) + partition(18) + residual(14x2) + hi(2) -> {os.path.join(OUT, tag)}')

    print('DONE ->', OUT)


if __name__ == '__main__':
    main()
