"""
exp_hs_monitor.py  —  Is UNSUPERVISED health-state monitoring effective?

Idea (CaMBN-style, role 2):  build a model of NORMAL sensor behaviour from the
run-in cycles only (assumed healthy -- the ground-truth hs label is NOT used to
build the detector), then declare 'abnormal' when the residual leaves the normal
band, via a residual T^2 statistic on an EWMA control chart.  Detection = first
EWMA crossing of the control limit.

Effectiveness is then judged against the ground-truth hs (A_dev[:,3]):
  - detection delay (predicted onset - true onset, in cycles)
  - false alarms (firing while still healthy)
  - per-cycle classification accuracy / precision / recall / F1
  - in-control mean / chart limit

Run:
  /opt/anaconda3/envs/pt_prac/bin/python3 exp_hs_monitor.py
"""
import os, sys, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_lib as L
from vecchia_gp import VecchiaGP

OUT = L.OUTPUT_NAMES


def per_cycle_groups(rows, cyc, n_per, seed):
    """Sample up to n_per rows from each cycle; return (cycle_list, idx_per_cycle)."""
    rng = np.random.default_rng(seed)
    ucyc = np.unique(cyc)
    sel = []
    for cc in ucyc:
        r = rows[cyc == cc]
        if len(r) > n_per:
            r = rng.choice(r, size=n_per, replace=False)
        sel.append(r)
    return ucyc, sel


def true_onset(hs_cyc, ucyc):
    below = np.where(hs_cyc < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--warmup', type=int, default=10)     # run-in cycles assumed healthy
    ap.add_argument('--n_per', type=int, default=30)      # within-cycle samples
    ap.add_argument('--seed_per', type=int, default=80)   # seed samples per warmup cycle
    ap.add_argument('--m', type=int, default=15)
    ap.add_argument('--iters', type=int, default=150)
    ap.add_argument('--lam', type=float, default=0.2)     # EWMA smoothing
    ap.add_argument('--K', type=float, default=3.0)       # control-limit width
    args = ap.parse_args()

    c = L.load_cache()
    W, X, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    u_col, c_col, hs_col = A[:, 0].astype(int), A[:, 1].astype(int), A[:, 3]
    units = np.unique(u_col)

    # ---- seed NORMAL model on run-in cycles (UNSUPERVISED: hs not used) ----
    seed_idx = []
    for u in units:
        rows = np.where(u_col == u)[0]
        cyc = c_col[rows]
        rng = np.random.default_rng(int(u) * 13)
        for cc in range(1, args.warmup + 1):
            r = rows[cyc == cc]
            if len(r):
                seed_idx.append(rng.choice(r, size=min(args.seed_per, len(r)), replace=False))
    seed_idx = np.concatenate(seed_idx)
    print(f'[seed] {len(seed_idx)} run-in points from first {args.warmup} cycles x {len(units)} units')

    vg = VecchiaGP(m=args.m).fit(W[seed_idx], X[seed_idx], iters=args.iters, verbose=False)

    # in-control residual mean/std (standardiser for the T^2 statistic)
    mean_s, _ = vg.predict(W[seed_idx])
    res_s = X[seed_idx] - mean_s
    mu_r, sd_r = res_s.mean(0), res_s.std(0) + 1e-8

    # ---- per-unit monitoring ----
    rows_per_unit = {}
    Tcurves = {}
    for u in units:
        rows = np.where(u_col == u)[0]
        cyc = c_col[rows]
        ucyc, sel = per_cycle_groups(rows, cyc, args.n_per, seed=int(u) * 91 + 7)
        allidx = np.concatenate(sel)
        mean_p, _ = vg.predict(W[allidx])
        z = (X[allidx] - mean_p - mu_r) / sd_r          # standardised residual
        t2_row = (z ** 2).sum(1)                        # Hotelling-style T^2 per row
        off = 0; Tc = []
        for r in sel:
            n = len(r); Tc.append(t2_row[off:off + n].mean()); off += n
        Tcurves[int(u)] = (ucyc, np.array(Tc))
        rows_per_unit[int(u)] = (rows, cyc)

    # ---- EWMA per unit (depends on lam only) + in-control limit base ----
    lam = args.lam
    ic_T = []
    ewma_store = {}
    truths = {}
    for u in units:
        ucyc, Tc = Tcurves[int(u)]
        ic_T.append(Tc[ucyc <= args.warmup])
        E = np.empty_like(Tc); e = mu0_seed = float(np.concatenate(ic_T).mean())
        for i, t in enumerate(Tc):
            e = lam * t + (1 - lam) * e
            E[i] = e
        ewma_store[int(u)] = (ucyc, E)
        rows, cyc = rows_per_unit[int(u)]
        hs_cyc = np.array([hs_col[rows[cyc == cc]].mean() for cc in ucyc])
        truths[int(u)] = true_onset(hs_cyc, ucyc)
    ic_T = np.concatenate(ic_T)
    mu0, sd0 = float(ic_T.mean()), float(ic_T.std() + 1e-8)
    # recompute EWMA with proper mu0 start
    for u in units:
        ucyc, Tc = Tcurves[int(u)]
        E = np.empty_like(Tc); e = mu0
        for i, t in enumerate(Tc):
            e = lam * t + (1 - lam) * e
            E[i] = e
        ewma_store[int(u)] = (ucyc, E)

    def evaluate(K):
        UCL = mu0 + K * sd0 * np.sqrt(lam / (2 - lam))
        rec = []
        for u in units:
            ucyc, E = ewma_store[int(u)]
            cross = np.where(E > UCL)[0]
            pred = int(ucyc[cross[0]]) if len(cross) else int(ucyc[-1] + 1)
            tru = truths[int(u)]
            pred_lab = (ucyc >= pred).astype(int)
            true_lab = (ucyc >= tru).astype(int)
            tp = int(((pred_lab == 1) & (true_lab == 1)).sum())
            fp = int(((pred_lab == 1) & (true_lab == 0)).sum())
            fn = int(((pred_lab == 0) & (true_lab == 1)).sum())
            tn = int(((pred_lab == 0) & (true_lab == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            recl = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0.0
            acc = (tp + tn) / len(ucyc)
            rec.append(dict(unit=int(u), true=tru, pred=pred, delay=pred - tru,
                            false_alarm=int(pred <= args.warmup),          # genuine: fired in run-in
                            early_warn=int(args.warmup < pred < tru),       # desirable lead
                            miss=int(not len(cross)),
                            prec=prec, recl=recl, f1=f1, acc=acc, ncyc=int(len(ucyc)), UCL=UCL))
        return rec

    # ---- K sweep : operating-point tradeoff ----
    print(f'[chart] in-control mu0={mu0:.2f} sd0={sd0:.2f}  (lam={lam})')
    print('\nK-sweep (control-limit width):  falseAlarm = fired within run-in (genuine);  '
          'earlyWarn = fired after run-in but before label (desirable lead)')
    print(f'{"K":>4} | {"UCL":>6} | {"meanDelay":>9} | {"|delay|":>7} | '
          f'{"falseAl":>7} | {"earlyWarn":>9} | {"missed":>6} | {"meanF1":>6} | {"meanAcc":>7}')
    print('-' * 88)
    sweeps = {}
    for K in [3, 4, 5, 6, 8, 10]:
        rec = evaluate(K); sweeps[K] = rec
        d = np.array([r['delay'] for r in rec])
        fa = sum(r['false_alarm'] for r in rec); ew = sum(r['early_warn'] for r in rec)
        ms = sum(r['miss'] for r in rec)
        print(f'{K:>4} | {rec[0]["UCL"]:>6.1f} | {d.mean():>+9.1f} | {np.abs(d).mean():>7.1f} | '
              f'{fa:>7} | {ew:>9} | {ms:>6} | {np.mean([r["f1"] for r in rec]):>6.3f} | '
              f'{np.mean([r["acc"] for r in rec]):>7.3f}')
    print('-' * 88)

    # operating point: best mean F1
    chosen = max([3, 4, 5, 6, 8, 10], key=lambda K: np.mean([r['f1'] for r in sweeps[K]]))
    rec = sweeps[chosen]
    print(f'\n[operating point]  K = {chosen}  (UCL={rec[0]["UCL"]:.1f}, chosen by best mean F1)')

    # ---- report ----
    print('\n' + '=' * 78)
    print(f'{"unit":>4} | {"ncyc":>4} | {"true":>4} | {"pred":>4} | {"delay":>5} | '
          f'{"falseAl":>7} | {"prec":>5} | {"recall":>6} | {"F1":>5} | {"acc":>5}')
    print('-' * 78)
    for r in rec:
        print(f'{r["unit"]:>4} | {r["ncyc"]:>4} | {r["true"]:>4} | {r["pred"]:>4} | '
              f'{r["delay"]:>5} | {r["false_alarm"]:>7} | {r["prec"]:>5.2f} | '
              f'{r["recl"]:>6.2f} | {r["f1"]:>5.2f} | {r["acc"]:>5.2f}')
    print('-' * 78)
    delays = np.array([r['delay'] for r in rec])
    fa = sum(r['false_alarm'] for r in rec)
    print(f'mean signed delay = {delays.mean():+.1f} cyc | mean |delay| = '
          f'{np.abs(delays).mean():.1f} cyc | false alarms = {fa}/{len(rec)} | '
          f'mean F1 = {np.mean([r["f1"] for r in rec]):.3f} | '
          f'mean acc = {np.mean([r["acc"] for r in rec]):.3f}')
    print('=' * 78)

    # ---- plot ----
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(units); cols = 3; rows_ = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_, cols, figsize=(5 * cols, 3 * rows_))
        axes = np.array(axes).reshape(-1)
        for ax, r in zip(axes, rec):
            u = r['unit']; ucyc, E = ewma_store[u]
            ax.plot(ucyc, E, color='#2f5c8f', lw=1.5, label='EWMA T²')
            ax.axhline(r['UCL'], color='#c0504d', ls='--', lw=1, label='UCL')
            ax.axvline(r['true'], color='#2e8b7f', ls='-', lw=1.4, label='true onset')
            ax.axvline(r['pred'], color='#c0504d', ls=':', lw=1.6, label='pred onset')
            ax.set_title(f'unit {u}: delay={r["delay"]:+d}, F1={r["f1"]:.2f}', fontsize=9)
            ax.set_xlabel('cycle'); ax.set_yscale('log'); ax.grid(alpha=0.3)
        for ax in axes[len(rec):]:
            ax.axis('off')
        axes[0].legend(fontsize=7, loc='upper left')
        plt.tight_layout()
        p = os.path.join(HERE, 'hs_monitor.png')
        plt.savefig(p, dpi=150); print('saved', p)
    except Exception as e:
        print('plot skipped:', e)


if __name__ == '__main__':
    main()
