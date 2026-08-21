"""
exp_rul_r23.py — R2 (HI rebuild + summary-variant selection) and
                 R3 (Stage-5 axis comparison: cycle vs cumulative flight-hours).

Everything dev-only (test tables in the R1 cache stay untouched until R5).

R2: for each seed and summary variant {mean, median, trim}:
      dev z-normalise -> HI MLP (frozen HI_CFG, input dim auto=5)
      -> HI shape check (start<=0.30, end>=0.93, last-decile monotone)
      -> selection among passing variants by dev LOO truncation-RUL NASA score
         (cycle axis, beta grid CV — the same criterion Stage 5 already uses)
R3: with the selected variant:
      (a) cycle axis  : LOO truncation RUL over dev units (exp basis, beta CV)
      (b) hours axis  : same, basis h(t)=g0+g2(e^{beta t}-1), t = cum hours/500;
                        RUL_hours -> cycles via mean observed flight length
      report RMSE / NASA per truncation fraction and overall.
"""
import os, sys, time, itertools
import numpy as np
from numpy.linalg import inv
from scipy import stats as sstats
import torch
torch.set_num_threads(int(os.environ.get('NTHREADS', 10)))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
VARIANTS = ['mean', 'median', 'trim']
HI_CFG = dict(epochs=1000, lambda0=1.0, lambda1=8.0, lambda2=2.0,
              init_threshold=0.2, flat_w=800.0, flat_m=0.004, alpha=0.001)
FRACS = [0.2, 0.4, 0.6, 0.8]
BETA_C = [16, 20, 25, 30, 35, 40, 45, 50]        # cycle axis (c = cyc/500)
BETA_T = [3, 4, 6, 8, 10, 12, 16, 20, 25]        # hours axis (t = hours/500)
GRID = np.arange(500) / 500.0
RES = os.path.join(HERE, 'rul_r23_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f:
        f.write(s + '\n')


def nasa(P, T):
    d = np.asarray(P, float) - np.asarray(T, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def load_dev(sd, variant):
    z = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k.split('_')[1]) for k in z.files if k.startswith('dev_')
                    and k.endswith('_ucyc')})
    raw = {u: z[f'dev_{u}_{variant}'] for u in units}
    hrs = {u: np.cumsum(z[f'dev_{u}_hours']) for u in units}
    allr = np.concatenate([raw[u] for u in units])
    mu, sg = allr.mean(0), allr.std(0) + 1e-8
    Z = {u: (raw[u] - mu) / sg for u in units}
    return units, Z, hrs


def fit_hi(sd, Z, units):
    np.random.seed(sd)
    him, _ = train_model_tail([Z[u] for u in units], **HI_CFG)
    return {u: him.forward(Z[u]).flatten() for u in units}


def shape_ok(HIs):
    starts = np.mean([h[:3].mean() for h in HIs.values()])
    ends = np.mean([h[-3:].mean() for h in HIs.values()])
    mono = np.mean([h[int(.9 * len(h)):].mean() >= h[int(.8 * len(h)):int(.9 * len(h))].mean()
                    for h in HIs.values()])
    return starts <= 0.30 and ends >= 0.93 and mono >= 0.5, (starts, ends, mono)


# ---------- generic exponential first-passage on arbitrary coordinates ----------
def eval_units(beta, coords_tr, hi_tr, coords_te, hi_te, fracs, to_cycles=None):
    """coords in [0,1] units of the 500-grid. to_cycles: None -> RUL in grid
    steps (cycle axis); else fn(u, cut, rul_grid_units) -> cycles."""
    psi = lambda t: np.column_stack([np.ones(len(t)), np.exp(beta * t) - 1])
    us = sorted(hi_tr)
    gam = np.array([np.linalg.lstsq(psi(coords_tr[u]), hi_tr[u], rcond=None)[0]
                    for u in us])
    s2 = np.mean([((hi_tr[u][3:] - (psi(coords_tr[u]) @ g)[3:]) ** 2).sum()
                  / max(len(hi_tr[u]) - 3, 1) for u, g in zip(us, gam)])
    mu0, cov0 = gam.mean(0), np.cov(gam.T) + 1e-10 * np.eye(2)
    PsiG = psi(GRID)

    P, T, FR, UN = [], [], [], []
    for f in fracs:
        for u in sorted(hi_te):
            h = hi_te[u]; n = len(h); cut = max(4, int(n * f))
            tc = coords_te[u][:cut]
            pm = psi(tc)
            An = inv((pm.T @ pm) / s2 + inv(cov0))
            mu = An @ ((pm.T @ h[:cut]) / s2 + inv(cov0) @ mu0)
            t_now = coords_te[u][cut - 1]
            i0 = int(np.searchsorted(GRID, t_now))
            fx = lambda pp: (pp @ mu - 1) / max(np.sqrt(max(pp @ An @ pp.T, 1e-12)), 1e-9)
            tmin = i0; pmin = 0
            for tt in range(i0, 500):
                pr = sstats.norm.cdf(fx(PsiG[tt]))
                if pmin < pr <= 0.5: pmin = pr; tmin = tt
                if pr > 0.9: break
            tmax = min(tmin + 1, 499); pmax = sstats.norm.cdf(fx(PsiG[tmax]))
            tstar = (tmax if pmax == pmin else
                     tmax - (tmax - tmin) * (pmax - 0.5) / (pmax - pmin))
            rul_grid = tstar / 500.0 - t_now                # in coord units
            if to_cycles is None:
                P.append(rul_grid * 500.0)
            else:
                P.append(to_cycles(u, cut, rul_grid))
            T.append(n - cut); FR.append(f); UN.append(u)
    return np.array(P), np.array(T), np.array(FR), np.array(UN)


def loo(axis, units, HIs, hrs, beta_grid, fracs=FRACS):
    """dev leave-one-unit-out truncation RUL. Returns pooled P,T and beta."""
    if axis == 'cycle':
        coords = {u: np.arange(len(HIs[u])) / 500.0 for u in units}
        to_cyc = None
    else:
        coords = {u: hrs[u] / 500.0 for u in units}
        def to_cyc(u, cut, rul_grid, _h=hrs):
            mean_fl = (_h[u][cut - 1] / cut) / 500.0        # mean flight, grid units
            return rul_grid / mean_fl
    # beta by nested LOO on NASA (same criterion as official select_beta)
    best = None
    for b in beta_grid:
        tot = 0.0
        for u in units:
            tr = [v for v in units if v != u]
            P, T, _, _ = eval_units(b, {v: coords[v] for v in tr},
                                    {v: HIs[v] for v in tr},
                                    {u: coords[u]}, {u: HIs[u]}, fracs, to_cyc)
            tot += nasa(P, T)
        if best is None or tot < best[1]:
            best = (b, tot)
    b = best[0]
    allP, allT, allF = [], [], []
    for u in units:
        tr = [v for v in units if v != u]
        P, T, F, _ = eval_units(b, {v: coords[v] for v in tr},
                                {v: HIs[v] for v in tr},
                                {u: coords[u]}, {u: HIs[u]}, fracs, to_cyc)
        allP.append(P); allT.append(T); allF.append(F)
    return np.concatenate(allP), np.concatenate(allT), np.concatenate(allF), b, best[1]


def main():
    open(RES, 'w').close()
    log(f'R2+R3  variants={VARIANTS}  HI_CFG frozen  dev-only LOO evaluation')
    t0 = time.time()
    chosen = {}
    HIstore = {}
    for sd in SEEDS:
        rows = []
        for v in VARIANTS:
            units, Z, hrs = load_dev(sd, v)
            HIs = fit_hi(sd, Z, units)
            ok, (st, en, mo) = shape_ok(HIs)
            P, T, F, b, sc = loo('cycle', units, HIs, hrs, BETA_C)
            rmse = float(np.sqrt(((P - T) ** 2).mean()))
            rows.append((v, ok, st, en, rmse, sc, HIs, hrs, units))
            log(f'  seed{sd} {v:>6}: shape={"OK " if ok else "FAIL"} '
                f'(start {st:.2f} end {en:.2f})  cycLOO RMSE={rmse:.2f} nasa={sc:.0f} '
                f'beta={b}  ({time.time()-t0:.0f}s)')
        passing = [r for r in rows if r[1]] or rows
        sel = min(passing, key=lambda r: r[5])
        chosen[sd] = sel[0]
        HIstore[sd] = (sel[8], sel[6], sel[7])
        log(f'  seed{sd} -> variant {sel[0]}')

    log('')
    log(f'{"axis":>7} | {"RMSE s0/s1/s2":>20} | {"RMSE mean":>9} | {"NASA mean":>9} | beta')
    log('-' * 72)
    summary = {}
    for axis, grid in [('cycle', BETA_C), ('hours', BETA_T)]:
        rs, scs, bs, perf = [], [], [], {f: [] for f in FRACS}
        for sd in SEEDS:
            units, HIs, hrs = HIstore[sd]
            P, T, F, b, _ = loo(axis, units, HIs, hrs, grid)
            rs.append(float(np.sqrt(((P - T) ** 2).mean())))
            scs.append(nasa(P, T)); bs.append(b)
            for f in FRACS:
                m = F == f
                perf[f].append(float(np.sqrt(((P[m] - T[m]) ** 2).mean())))
        summary[axis] = (rs, scs, perf)
        log(f'{axis:>7} | {rs[0]:>6.2f} {rs[1]:>6.2f} {rs[2]:>6.2f} | '
            f'{np.mean(rs):>9.2f} | {np.mean(scs):>9.1f} | {bs}')
    log('')
    log('per-truncation RMSE (mean over seeds):')
    log(f'{"frac":>6} | {"cycle":>7} | {"hours":>7}')
    for f in FRACS:
        log(f'{f:>6.0%} | {np.mean(summary["cycle"][2][f]):>7.2f} | '
            f'{np.mean(summary["hours"][2][f]):>7.2f}')
    log(f'\nchosen variants: {chosen}')
    log(f'wall={((time.time()-t0)/60):.1f} min')


if __name__ == '__main__':
    main()
