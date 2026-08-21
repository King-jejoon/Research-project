"""
exp_v3_d_hi.py — STAGE D (core): HI network — lambda1 (monotonicity) x
lambda2 (convexity) grid, justified TABLE-IV style by HI property metrics.

Frozen at the previous chain's values: initial level (lambda0=1, thr=0.2),
end level (end_target=1.03), tail flatness (w=800, m=0.004), Adam 1e-3,
1000 epochs.  Tuned: lambda1 in {2,4,8,16} x lambda2 in {0.5,1,2,4}.

Inputs: per-flight trim25 of gate-passing residuals (winner set SET, gate V*
at pooled percentile VP from Stage B), dev z-normalized.  3 seeds.

Property metrics per (l1,l2), averaged over units and seeds
(HI resampled to a common normalized-life grid where needed):
  Monotonicity : sum of negative increments, x100      (0 = perfectly monotone)
  Curvature    : sum of second differences, x100       (>0 = accelerating/convex)
  Info range   : mean(end HI) - mean(start HI)         (dynamic range, ~1 ideal)
  Dist shift   : std of HI across units on the common life grid, averaged
                 (unit-to-unit consistency; lower = better)
Reference column: dev LOO truncation RUL RMSE (cycle axis, beta by nested LOO
NASA — the frozen rule).  Shape gate (start<=0.30, end>=0.93, tail monotone)
must pass.
Selection: among shape-passing combos, maximize the composite of the four
properties (each min-max normalized over the grid with the proper sign);
the LOO RUL column verifies the property-based choice is not perverse.
Outputs: TABLE-IV-style table, best HI curves figure, selected (l1,l2).
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_rul_r23 import (HI_CFG, BETA_C, nasa, loo, train_model_tail, shape_ok)
from exp_v3_a2cmp import load as load_stats, SEEDS, FILES

SET = int(os.environ.get('SET', 5))
VP = int(os.environ.get('VP', 25))
L1S = [2.0, 4.0, 8.0, 16.0]
L2S = [0.5, 1.0, 2.0, 4.0]
RES = os.path.join(HERE, 'v3_d_hi_results.txt')
OUT = __import__('exp_paths').PAPER


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def trim25(v):
    s = np.sort(v, axis=0); n = len(v)
    return s[int(.25 * n):max(int(.25 * n) + 1, int(.75 * n))].mean(0)


def build_tables(sd, V):
    """per-unit z-normalized trim25 summaries + cumulative hours."""
    Z = np.load(os.path.join(HERE, FILES[SET].format(sd=sd)))
    H = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k[1:].split('_')[0]) for k in Z.files if k.endswith('_cc')})
    raw, hrs = {}, {}
    for u in units:
        cc = Z[f'u{u}_cc']; rs = Z[f'u{u}_resid']; dc = Z[f'u{u}_dc']
        ur = H[f'dev_{u}_ucyc']; durs = H[f'dev_{u}_hours']
        pos = {int(c): i for i, c in enumerate(ur)}
        ucyc = np.array([c for c in np.unique(cc) if int(c) in pos])
        T = np.empty((len(ucyc), rs.shape[1]))
        for i, c in enumerate(ucyc):
            b = cc == c
            m = b & (dc >= V)
            T[i] = trim25(rs[m if m.any() else b])
        raw[u] = T
        hrs[u] = np.cumsum(np.array([durs[pos[int(c)]] for c in ucyc], float))
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Zn = {u: (raw[u] - mu) / sg for u in units}
    return units, Zn, hrs


def properties(HIs):
    """TABLE-IV style property metrics for one seed's HI set."""
    mon, curv, rng_ = [], [], []
    G = np.linspace(0, 1, 101)
    interp = []
    for h in HIs.values():
        d1 = np.diff(h)
        mon.append(100.0 * np.minimum(d1, 0).sum())
        curv.append(100.0 * np.diff(h, 2).sum())
        rng_.append(h[-3:].mean() - h[:3].mean())
        x = np.linspace(0, 1, len(h))
        interp.append(np.interp(G, x, h))
    shift = float(np.std(np.stack(interp), axis=0).mean())
    return (float(np.mean(mon)), float(np.mean(curv)),
            float(np.mean(rng_)), shift)


def main():
    open(RES, 'w').close()
    t00 = time.time()
    log(f'STAGE D — HI lambda1 x lambda2 grid  (set{SET}, gate V@p{VP}, trim25)')
    log('  frozen: lambda0=1, thr=0.2, end_target=1.03, flat=(800,0.004), 1e-3x1000ep')

    Vs = {}
    data = {}
    for sd in SEEDS:
        D = load_stats(SET, sd)
        Vs[sd] = float(np.percentile(
            np.concatenate([D[u]['dc'] for u in D]), VP))
        data[sd] = build_tables(sd, Vs[sd])
    log(f'  V* per seed: {[round(Vs[s],2) for s in SEEDS]}')

    R = {}
    for l1 in L1S:
        for l2 in L2S:
            cfg = dict(HI_CFG); cfg.update(lambda1=l1, lambda2=l2,
                                           end_target=1.03)
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
                               rul=float(np.mean(rmses)),
                               rul_sd=float(np.std(rmses)),
                               shape=int(np.sum(oks)))
            log(f'  l1={l1:>4} l2={l2:>3}: Mon={pr[0]:8.2f}  Curv={pr[1]:7.2f}  '
                f'Range={pr[2]:5.3f}  Shift={pr[3]:6.4f}  '
                f'LOO RUL={np.mean(rmses):.2f}±{np.std(rmses):.2f}  '
                f'shape {int(np.sum(oks))}/3')

    # ---- TABLE IV style ----
    log('')
    log('TABLE (rows lambda1; column groups lambda2; cells: Mon | Curv | Range | Shift)')
    hdr = '  l1\\l2 |' + '|'.join(f'   {l2:^37}' for l2 in L2S)
    log(hdr)
    for l1 in L1S:
        cells = []
        for l2 in L2S:
            r = R[(l1, l2)]
            cells.append(f'{r["mon"]:8.2f} {r["curv"]:7.2f} {r["rng"]:5.3f} '
                         f'{r["shift"]:6.4f}')
        log(f'  {l1:>5} |' + '|'.join(f' {c} ' for c in cells))

    # ---- selection: shape pass, then composite of normalized properties ----
    ok_keys = [k for k in R if R[k]['shape'] == 3]
    pool_keys = ok_keys if ok_keys else list(R)
    arr = {p: np.array([R[k][p] for k in pool_keys])
           for p in ['mon', 'curv', 'rng', 'shift']}

    def norm(v, sign):
        v = sign * v
        return (v - v.min()) / (v.max() - v.min() + 1e-12)

    comp = (norm(arr['mon'], +1) + norm(arr['curv'], +1) +
            norm(arr['rng'], +1) + norm(arr['shift'], -1)) / 4
    sel = pool_keys[int(np.argmax(comp))]
    log('')
    log(f'shape-passing combos: {sorted(ok_keys)}')
    log(f'SELECTED (property composite): lambda1={sel[0]}, lambda2={sel[1]}  '
        f'composite={comp.max():.3f}')
    log(f'  its LOO RUL: {R[sel]["rul"]:.2f} ± {R[sel]["rul_sd"]:.2f}  '
        f'(grid best RUL: {min(R.values(), key=lambda r: r["rul"])["rul"]:.2f})')

    # ---- best HI curves figure (seed 0) ----
    cfg = dict(HI_CFG); cfg.update(lambda1=sel[0], lambda2=sel[1], end_target=1.03)
    units, Zn, hrs = data[0]
    np.random.seed(0)
    him, _ = train_model_tail([Zn[u] for u in units], **cfg)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for u in units:
        h = him.forward(Zn[u]).flatten()
        ax.plot(np.arange(len(h)), h, lw=1.4, label=f'unit {u}')
    ax.set_xlabel('Flight cycle'); ax.set_ylabel('Health Index')
    ax.set_title(f'HI trajectories (dev units, seed 0; '
                 f'lambda1={sel[0]:g}, lambda2={sel[1]:g})')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=7, ncol=3, frameon=False)
    fig.tight_layout()
    p_fig = os.path.join(OUT, 'fig_v3_hi_curves.png')
    fig.savefig(p_fig, dpi=150); plt.close(fig)
    log(f'figure: {p_fig}')
    np.savez(os.path.join(HERE, 'v3_d_hi.npz'),
             l1=np.array([sel[0]]), l2=np.array([sel[1]]),
             **{f'{k[0]}_{k[1]}_{p}': R[k][p] for k in R
                for p in ['mon', 'curv', 'rng', 'shift', 'rul']})
    log(f'\nwall={(time.time()-t00)/60:.1f} min')


if __name__ == '__main__':
    main()
