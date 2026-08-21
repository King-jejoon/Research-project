"""
exp_ll_detect.py — resurrect & improve the colleague's condition-likelihood
detection (paper method #2), after fixing the whitening bug (see demo_cond2.py).

Detection variants, all on the SAME EWMA control chart machinery and the SAME
sampling as exp_final Stage-1 (NPER=60, warmup 10, lam 0.2, seeds 0/1/2):

  llorig : colleague's original conditional LL (buggy quad)   -> LCL (LL drops)
  llfix  : corrected conditional LL                           -> LCL
  m2     : Mahalanobis^2 = r^T S^{-1} r  (uncertainty-whitened
           residual, ~chi2(q) in control; the 'condition
           likelihood' statistic with the detcov term removed) -> UCL (rises)

Normal model = colleague's GPyTorchMOGP (Vecchia-trained) fit ONCE per seed on
pooled run-in points of all dev units (cheap; avoids v2's per-unit fits that
took hours).  Per-cycle statistic = mean over the cycle's sampled points.

Scored vs ground-truth hs exactly like exp_detect_rul.py (acc / F1 / delay).
K is swept post-hoc on the stored per-cycle curves; best-F1 operating point
reported per variant.  Baseline to beat: T^2+EWMA = 80.7% acc / F1 0.866.

Run:  /opt/anaconda3/envs/pt_prac/bin/python3 exp_ll_detect.py
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond2 import conditional_stats2

DT = torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, LAM, NPER = 10, 0.2, 60
MCOND = 15                     # neighbours for conditional stats
RUNIN_PER_UNIT = 250           # run-in points per unit for the global normal model
SEEDS = [0, 1, 2]
KSWEEP = [3, 4, 5, 6, 8, 10]
RES = os.path.join(HERE, 'll_detect_results.txt')


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def groups_of(rows, cyc, n_per, seed):
    rng = np.random.default_rng(seed); out = []
    for cc in np.unique(cyc):
        r = rows[cyc == cc]
        if len(r) > n_per: r = rng.choice(r, n_per, replace=False)
        out.append(r)
    return np.unique(cyc), out


def fit_mogp(Wtr, Ytr, m=18, steps=150):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT); tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=Ytr.shape[1], rank=2, kernel='rbf').to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Ytr.shape[1]).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=0.05, group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


def true_onset(hs_by_cycle, ucyc):
    below = np.where(hs_by_cycle < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def ewma_detect(ucyc, stat, K, side):
    """EWMA chart on per-cycle statistic; side='up' -> UCL, 'down' -> LCL."""
    ic = stat[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
    lim = K * sd0 * np.sqrt(LAM / (2 - LAM))
    e = mu0; E = []
    for v in stat: e = LAM * v + (1 - LAM) * e; E.append(e)
    E = np.array(E)
    cross = np.where(E > mu0 + lim)[0] if side == 'up' else np.where(E < mu0 - lim)[0]
    return int(ucyc[cross[0]]) if len(cross) else int(ucyc[-1] + 1)


def score_unit(ucyc, trans, tru):
    pred_lab = (ucyc >= trans).astype(int); true_lab = (ucyc >= tru).astype(int)
    acc = float((pred_lab == true_lab).mean())
    tp = int(((pred_lab == 1) & (true_lab == 1)).sum()); fp = int(((pred_lab == 1) & (true_lab == 0)).sum())
    fn = int(((pred_lab == 0) & (true_lab == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; recl = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0.0
    return acc, f1


def main():
    open(RES, 'w').close()
    c = L.load_cache()
    W, Xall, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    X = Xall[:, SIDX]
    du = np.unique(A[:, 0].astype(int))
    t0 = time.time()
    log(f'LL-DETECT (colleague code, fixed)  seeds={SEEDS}  NPER={NPER}  MCOND={MCOND}  '
        f'runin/unit={RUNIN_PER_UNIT}')
    log(f'baseline to beat: T^2+EWMA = 80.7% acc / F1 0.866 / |delay| 14.6')

    # curves[seed][unit] = (ucyc, {'llorig':..., 'llfix':..., 'm2':...}, true_onset)
    curves = {}
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        # ---- global normal model on pooled run-in (colleague's MOGP) ----
        ridx = []
        for u in du:
            rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
            rin = rows[cyc <= WARMUP]
            rng = np.random.default_rng(int(u) * 13 + sd)
            ridx.append(rng.choice(rin, min(RUNIN_PER_UNIT, len(rin)), replace=False))
        ridx = np.concatenate(ridx)
        gp = fit_mogp(W[ridx], X[ridx], m=18, steps=150)
        log(f'  seed{sd}: normal MOGP fit on {len(ridx)} run-in pts ({time.time()-t0:.0f}s)')

        curves[sd] = {}
        for u in du:
            rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + sd)
            idx = np.concatenate(sel)
            cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
            teX = torch.tensor(gp['xs'].transform(W[idx]), dtype=DT)
            teY = torch.tensor(gp['ys'].transform(X[idx]), dtype=DT)
            _, detcov, m2, llfix, llorig = conditional_stats2(
                gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'], teX,
                m=MCOND, test_y=teY)
            stats_c = {}
            for name, v in [('llorig', llorig), ('llfix', llfix), ('m2', m2)]:
                stats_c[name] = np.array([v[cc == kk].mean() for kk in ucyc])
            hs_by_cycle = np.array([A[rows[cyc == cc0], 3].mean() for cc0 in ucyc])
            tru = true_onset(hs_by_cycle, ucyc)
            curves[sd][int(u)] = (ucyc, stats_c, tru)
            log(f'    unit {u}: cycles={len(ucyc)} true={tru} ({time.time()-t0:.0f}s)')

    # ---- post-hoc K sweep per variant ----
    SIDE = {'llorig': 'down', 'llfix': 'down', 'm2': 'up'}
    log('\n' + '=' * 78)
    log('K-SWEEP  (mean over 9 units x 3 seeds; acc% / F1 / mean|delay|)')
    log('=' * 78)
    hdr = f'{"K":>4} |'
    for v in SIDE: hdr += f'  {v:>22} |'
    log(hdr)
    best = {}
    for K in KSWEEP:
        row = f'{K:>4} |'
        for v in SIDE:
            accs, f1s, dls = [], [], []
            for sd in SEEDS:
                for u in curves[sd]:
                    ucyc, sc, tru = curves[sd][u]
                    trans = ewma_detect(ucyc, sc[v], K, SIDE[v])
                    a, f = score_unit(ucyc, trans, tru)
                    accs.append(a); f1s.append(f); dls.append(abs(trans - tru))
            ma, mf, md = np.mean(accs) * 100, np.mean(f1s), np.mean(dls)
            row += f'  {ma:5.1f}% {mf:.3f} {md:5.1f}cyc |'
            if v not in best or mf > best[v][1]:
                best[v] = (K, mf, ma, md)
        log(row)
    log('-' * 78)
    for v in SIDE:
        K, mf, ma, md = best[v]
        log(f'best {v:>7}: K={K}  acc={ma:.1f}%  F1={mf:.3f}  |delay|={md:.1f}cyc')
    log(f'baseline T^2: K=5  acc=80.7%  F1=0.866  |delay|=14.6cyc')

    # ---- per-unit detail at each variant's best K ----
    for v in SIDE:
        K = best[v][0]
        log('\n' + '=' * 78)
        log(f'PER-UNIT @ best K={K}  —  variant {v}')
        log(f'{"unit":>4} | {"true":>4} | {"pred s0/s1/s2":>14} | {"acc%":>6} | {"F1":>5}')
        log('-' * 78)
        for u in sorted(curves[SEEDS[0]]):
            preds, accs, f1s = [], [], []
            for sd in SEEDS:
                ucyc, sc, tru = curves[sd][u]
                trans = ewma_detect(ucyc, sc[v], K, SIDE[v])
                a, f = score_unit(ucyc, trans, tru)
                preds.append(trans); accs.append(a); f1s.append(f)
            log(f'{u:>4} | {tru:>4} | {str(preds):>14} | {np.mean(accs)*100:>6.1f} | {np.mean(f1s):>5.2f}')
    log(f'\nwall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
