"""
exp_detect_rul.py  —  fresh run of the FINAL pipeline that ALSO scores the
Stage-1 transition detector against ground-truth hs, and reports RUL at the
truncation fractions requested: 20 / 40 / 60 / 80 %.

Two things it answers:
  (1) transition-detection ACCURACY (%) of the detector *as actually used in the
      pipeline* (NPER, K, per-unit seed identical to exp_final).  Scored on dev
      units against ground-truth hs (A_dev[:,3]) -- can be checked BEFORE RUL.
  (2) RUL RMSE using 20/40/60/80 % of each test HI curve (truncation eval).

Usage:
  /opt/anaconda3/envs/pt_prac/bin/python3 exp_detect_rul.py --detect-only   # fast, detection accuracy only
  /opt/anaconda3/envs/pt_prac/bin/python3 exp_detect_rul.py                  # full: detection + RUL
Config: CPU, float32, 6 threads.
"""
import os, sys, time, argparse
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
from scipy import stats
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from vecchia_gp import VecchiaGP
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp, vecchia_mogp_predictive_mean
import neural_fusion as NF

DT = torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, K, LAM, NPER = 10, 5.0, 0.2, 60
L0, L1, L2, INIT_THR, EPOCHS = 1.0, 6.0, 2.0, 0.2, 1000
SEEDS = [0, 1, 2]
FRACS = [0.2, 0.4, 0.6, 0.8]          # <-- requested truncation fractions
RES = os.path.join(HERE, 'detect_rul_results.txt')


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


def pmean(gp, Wq):
    q = torch.tensor(gp['xs'].transform(Wq), dtype=DT)
    with torch.no_grad():
        mu = vecchia_mogp_predictive_mean(gp['model'], gp['lik'], gp['tX'], gp['tY'], q, gp['struct'], m=18)
    return mu.cpu().numpy() * gp['ys'].scale_ + gp['ys'].mean_


def rul_of(devHI, testHI, fracs):
    t = np.arange(500).reshape(-1, 1) / 500; Psi = np.hstack((np.ones((500, 1)), t, t**2))
    Yd = [devHI[u] for u in sorted(devHI)]; ntr = len(Yd)
    gam = np.zeros((ntr, 3))
    for i, y in enumerate(Yd):
        n = len(y); Xp = Psi[:n]; gam[i] = (inv(Xp.T@Xp)@Xp.T@y).T
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gam[i])[3:]) ** 2).sum() / (len(Yd[i]) - 4) for i in range(ntr)])
    mu0, cov0 = gam.mean(0), np.cov(gam.T)
    def rul(HI):
        p = Psi[:len(HI)]; mu = inv((p.T@p)/s2 + inv(cov0)) @ ((p.T@HI)/s2 + inv(cov0)@mu0)
        cov = inv((p.T@p)/s2 + inv(cov0)); fx = lambda pp: (pp@mu - 1) / pow(pp@cov@pp.T, 0.5)
        tmin = len(HI)-1; pmin = 0
        for tt in range(len(HI)-1, 500):
            pr = stats.norm.cdf(fx(Psi[tt]))
            if pr > pmin and pr <= 0.5: pmin = pr; tmin = tt
            if pr > 0.9: break
        tmax = tmin+1; pmax = stats.norm.cdf(fx(Psi[tmax]))
        return (tmax if pmax == pmin else tmax - (tmax-tmin)*(pmax-0.5)/(pmax-pmin)) - len(HI) + 1
    per_f = {}
    P, T, FR = [], [], []
    for f in fracs:
        pf, tf = [], []
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n*f)); pf.append(rul(h[:cut])); tf.append(n - cut)
        per_f[f] = float(np.sqrt(((np.array(pf)-np.array(tf))**2).mean()))
        P += pf; T += tf; FR += [f] * len(pf)
    rul_of.last_PT = (np.array(P), np.array(T), np.array(FR))   # for score / saving
    return float(np.sqrt(((np.array(P)-np.array(T))**2).mean())), per_f


def true_onset(hs_by_cycle, ucyc):
    below = np.where(hs_by_cycle < 0.5)[0]
    return int(ucyc[below[0]]) if len(below) else int(ucyc[-1] + 1)


def detect(cache, sd):
    """Replicate exp_final Stage-1 EXACTLY and score each dev unit vs ground-truth hs.
    Returns (normal_idx, gp-inputs) plus per-unit detection records."""
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    X = Xall[:, SIDX]
    du = np.unique(A[:, 0].astype(int))
    normal_idx = []; recs = []
    for u in du:
        rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
        ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + sd)
        idx = np.concatenate(sel); cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
        rin = idx[cc <= WARMUP]
        vg = VecchiaGP(m=15).fit(W[rin], X[rin], iters=100, verbose=False)
        ms, _ = vg.predict(W[rin]); r0 = X[rin] - ms; mu_r, sd_r = r0.mean(0), r0.std(0) + 1e-8
        mp, _ = vg.predict(W[idx]); z = (X[idx] - mp - mu_r) / sd_r; t2 = (z ** 2).sum(1)
        uc = np.unique(cc); Tc = np.array([t2[cc == kk].mean() for kk in uc])
        ic = Tc[uc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
        UCL = mu0 + K * sd0 * np.sqrt(LAM / (2 - LAM)); e = mu0; E = []
        for v in Tc: e = LAM * v + (1 - LAM) * e; E.append(e)
        cr = np.where(np.array(E) > UCL)[0]; trans = int(uc[cr[0]]) if len(cr) else int(uc[-1] + 1)
        # ---- score against ground truth hs ----
        hs_by_cycle = np.array([A[rows[cyc == cc0], 3].mean() for cc0 in uc])
        tru = true_onset(hs_by_cycle, uc)
        pred_lab = (uc >= trans).astype(int); true_lab = (uc >= tru).astype(int)
        acc = float((pred_lab == true_lab).mean())
        tp = int(((pred_lab == 1) & (true_lab == 1)).sum()); fp = int(((pred_lab == 1) & (true_lab == 0)).sum())
        fn = int(((pred_lab == 0) & (true_lab == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; recl = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * recl / (prec + recl) if prec + recl else 0.0
        recs.append(dict(unit=int(u), true=int(tru), pred=int(trans), delay=int(trans - tru),
                         acc=acc, f1=f1, ncyc=int(len(uc))))
        normal_idx.append(idx[cc < trans])
    normal_idx = np.concatenate(normal_idx)
    return normal_idx, recs


def run_seed(sd, cache, detect_only=False):
    torch.manual_seed(sd); np.random.seed(sd)
    normal_idx, recs = detect(cache, sd)
    if detect_only:
        return None, None, recs

    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    Wt, Xtall, At = cache['W_test'], cache['X_s_test'], cache['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))
    rng = np.random.default_rng(sd)
    if len(normal_idx) > 5000: normal_idx = rng.choice(normal_idx, 5000, replace=False)

    gp = fit_mogp(W[normal_idx], X[normal_idx], m=18, steps=150)

    def resid(Wd, Xd, Ad, units):
        out = {}
        for u in units:
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 5 + sd)
            idx = np.concatenate(sel); cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
            pm = pmean(gp, Wd[idx]); r = Xd[idx] - pm
            d = {s: {} for s in SENS}
            for kk in np.unique(cc):
                mv = r[cc == kk].mean(0)
                for j, s in enumerate(SENS): d[s][int(kk)] = float(mv[j])
            out[int(u)] = d
        return out
    dev_res = resid(W, X, A, du); test_res = resid(Wt, Xt, At, tu)
    st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
              np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in SENS}
    nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
    dev_n, test_n = nrm(dev_res), nrm(test_res)

    def blist(dn):
        out = []
        for u in sorted(dn):
            cy = sorted(dn[u][SENS[0]].keys()); out.append((u, cy, np.array([[dn[u][s][k] for s in SENS] for k in cy])))
        return out
    dl, tl = blist(dev_n), blist(test_n)
    np.random.seed(sd)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=EPOCHS, lambda0=L0, lambda1=L1, lambda2=L2,
                            init_threshold=INIT_THR, alpha=0.001, verbose=False)
    dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
    tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
    rmse, per_f = rul_of(dHI, tHI, FRACS)
    return rmse, per_f, recs


def report_detection(all_recs):
    log('\n' + '=' * 74)
    log('STAGE-1 TRANSITION DETECTION ACCURACY  (dev units, vs ground-truth hs)')
    log('=' * 74)
    # per-unit averaged over seeds
    units = sorted({r['unit'] for recs in all_recs for r in recs})
    log(f'{"unit":>4} | {"true":>4} | {"pred(mean)":>10} | {"delay":>6} | {"acc%":>6} | {"F1":>5}')
    log('-' * 74)
    for u in units:
        rs = [r for recs in all_recs for r in recs if r['unit'] == u]
        pm = np.mean([r['pred'] for r in rs]); tr = rs[0]['true']
        dl = np.mean([r['delay'] for r in rs]); ac = np.mean([r['acc'] for r in rs]) * 100
        f1 = np.mean([r['f1'] for r in rs])
        log(f'{u:>4} | {tr:>4} | {pm:>10.1f} | {dl:>+6.1f} | {ac:>6.1f} | {f1:>5.2f}')
    log('-' * 74)
    flat = [r for recs in all_recs for r in recs]
    macc = np.mean([r['acc'] for r in flat]) * 100
    mf1 = np.mean([r['f1'] for r in flat])
    mad = np.mean([abs(r['delay']) for r in flat])
    log(f'OVERALL detection accuracy = {macc:.1f}%   mean F1 = {mf1:.3f}   mean |delay| = {mad:.1f} cyc')
    # per-seed accuracy
    for sd, recs in zip(SEEDS, all_recs):
        a = np.mean([r['acc'] for r in recs]) * 100
        log(f'  seed{sd}: detection acc = {a:.1f}%   transitions = {[r["pred"] for r in recs]}')
    log('=' * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detect-only', action='store_true')
    args = ap.parse_args()
    open(RES, 'w').close()
    cache = L.load_cache(); t0 = time.time()
    mode = 'DETECT-ONLY' if args.detect_only else 'FULL (detect + RUL)'
    log(f'{mode}  seeds={SEEDS}  NPER={NPER}  K={K}  fracs={FRACS}')

    all_recs = []; rmses = []; per_fs = []; allP, allT, allFR = [], [], []
    for sd in SEEDS:
        out = run_seed(sd, cache, detect_only=args.detect_only)
        if args.detect_only:
            _, _, recs = out; all_recs.append(recs)
            a = np.mean([r['acc'] for r in recs]) * 100
            log(f'  seed{sd}: detection acc = {a:.1f}%  ({time.time()-t0:.0f}s)')
        else:
            rmse, per_f, recs = out; all_recs.append(recs); rmses.append(rmse); per_fs.append(per_f)
            P, T, FR = rul_of.last_PT
            allP.append(P); allT.append(T); allFR.append(FR)
            pf = ' '.join(f'{f:.0%}:{v:.2f}' for f, v in per_f.items())
            log(f'  seed{sd}: RUL RMSE = {rmse:.2f}  (by trunc {pf})  ({time.time()-t0:.0f}s)')

    report_detection(all_recs)

    if not args.detect_only:
        def nasa_score(P, T):
            d = np.asarray(P, float) - np.asarray(T, float)
            return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
        np.savez(os.path.join(HERE, 'detect_rul_pred.npz'),
                 P=np.stack(allP), T=np.stack(allT), FR=np.stack(allFR), seeds=np.array(SEEDS))
        rmses = np.array(rmses)
        log('\n' + '=' * 74)
        log('RUL PREDICTION ACCURACY  (truncation eval, RMSE in cycles + NASA score)')
        log('=' * 74)
        log(f'{"data used":>10} | {"seed0":>7} | {"seed1":>7} | {"seed2":>7} | {"mean":>7} | {"std":>6} | {"score":>8}')
        log('-' * 74)
        for f in FRACS:
            vs = np.array([pf[f] for pf in per_fs])
            scs = np.mean([nasa_score(P[FR == f], T[FR == f]) for P, T, FR in zip(allP, allT, allFR)])
            log(f'{f:>9.0%} | {vs[0]:>7.2f} | {vs[1]:>7.2f} | {vs[2]:>7.2f} | {vs.mean():>7.2f} | {vs.std():>6.2f} | {scs:>8.1f}')
        log('-' * 74)
        scores = np.array([nasa_score(P, T) for P, T in zip(allP, allT)])
        log(f'{"OVERALL":>10} | {rmses[0]:>7.2f} | {rmses[1]:>7.2f} | {rmses[2]:>7.2f} | {rmses.mean():>7.2f} | {rmses.std():>6.2f} | {scores.mean():>8.1f}')
        log(f'NASA score per seed = {[f"{s:.1f}" for s in scores]}  mean = {scores.mean():.1f} +/- {scores.std():.1f}')
        log('=' * 74)
        log(f'wall time = {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
