"""
Uncertainty-filter ABLATION.  Does the detcov filter actually help RUL?

Filter can act at two places:
  RAW      : drop extreme-operating-parameter points before Stage 1 (reference GP)
  RESIDUAL : drop high-uncertainty points inside Stage 3 (Stage-2 GP)

4 configs x N seeds, everything else fixed per seed:
  none | raw | resid | raw+resid   -> mean truncation RUL RMSE (+/- std)
Also saves HI curves for the raw+resid config (hi_curves_ablation.png).
Config: CPU, float32, 6 threads.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from vecchia_gp import VecchiaGP
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp, vecchia_mogp_predictive_mean
from demo_cond import conditional_stats
import neural_fusion as NF

DT = torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, K, LAM, NPER, DROP = 10, 5.0, 0.2, 20, 0.15
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'ablation_results.txt')


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


def fit_mogp(Wtr, Ytr, m=18, steps=120):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT); tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=Ytr.shape[1], rank=2, kernel='rbf').to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Ytr.shape[1]).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=0.05, group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


def cond(gp, Wq, Yq=None):
    teX = torch.tensor(gp['xs'].transform(Wq), dtype=DT)
    teY = None if Yq is None else torch.tensor(gp['ys'].transform(Yq), dtype=DT)
    pred, detcov, _ = conditional_stats(gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'], teX, m=18, test_y=teY)
    return pred * gp['ys'].scale_ + gp['ys'].mean_, detcov


def pmean(gp, Wq):
    q = torch.tensor(gp['xs'].transform(Wq), dtype=DT)
    with torch.no_grad():
        mu = vecchia_mogp_predictive_mean(gp['model'], gp['lik'], gp['tX'], gp['tY'], q, gp['struct'], m=18)
    return mu.cpu().numpy() * gp['ys'].scale_ + gp['ys'].mean_


def rul_rmse(devHI, testHI):
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
    P, T = [], []
    for f in [0.5, 0.6, 0.7, 0.8]:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n*f)); P.append(rul(h[:cut])); T.append(n - cut)
    return float(np.sqrt(((np.array(P)-np.array(T))**2).mean()))


def build_hi(dev_n, test_n, seed):
    def blist(dn):
        out = []
        for u in sorted(dn):
            cy = sorted(dn[u][SENS[0]].keys()); out.append((u, cy, np.array([[dn[u][s][k] for s in SENS] for k in cy])))
        return out
    dl, tl = blist(dev_n), blist(test_n)
    np.random.seed(seed)
    him, _ = NF.train_model([d for _, _, d in dl], epochs=1000, lambda0=1.0, lambda1=6.0, lambda2=2.0,
                            init_threshold=0.2, alpha=0.001, verbose=False)
    dHI = {u: him.forward(d).flatten() for u, cy, d in dl}
    tHI = {u: him.forward(d).flatten() for u, cy, d in tl}
    return dHI, tHI, dl, tl


def main():
    open(RES, 'w').close()
    log(f'ABLATION  seeds={SEEDS}  NPER={NPER}  drop={DROP}')
    c = L.load_cache()
    W, Xall, A = c['W_dev'], c['X_s_dev'], c['A_dev']; Wt, Xtall, At = c['W_test'], c['X_s_test'], c['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))
    results = {k: [] for k in ['none', 'raw', 'resid', 'raw+resid']}
    saved_plot = False

    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        # reference GP + sampling + raw detcov (once per seed)
        seed_pts = []
        for u in du:
            rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
            s = rows[cyc <= WARMUP]; rng = np.random.default_rng(int(u) + sd)
            if len(s) > 160: s = rng.choice(s, 160, replace=False)
            seed_pts.append(s)
        gp0 = fit_mogp(W[np.concatenate(seed_pts)], X[np.concatenate(seed_pts)])
        def sample(Wd, Xd, Ad, u):
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + sd)
            idx = np.concatenate(sel); ccarr = np.concatenate([np.full(len(g), cc) for cc, g in zip(ucyc, sel)])
            _, dc = cond(gp0, Wd[idx], Xd[idx]); return idx, ccarr, dc
        dev_s = {int(u): sample(W, X, A, u) for u in du}; test_s = {int(u): sample(Wt, Xt, At, u) for u in tu}
        all_dc = np.concatenate([dev_s[u][2] for u in dev_s] + [test_s[u][2] for u in test_s])
        thr = np.percentile(all_dc, DROP * 100)

        for raw in [False, True]:
            # detection + Stage 2 (depend on raw filter)
            normal_idx = []
            for u in du:
                idx, cc, dc = dev_s[int(u)]
                if raw: m = dc >= thr; idx, cc = idx[m], cc[m]
                rin = idx[cc <= WARMUP]
                vg = VecchiaGP(m=15).fit(W[rin], X[rin], iters=100, verbose=False)
                ms, _ = vg.predict(W[rin]); r0 = X[rin] - ms; mu_r, sd_r = r0.mean(0), r0.std(0) + 1e-8
                mp, _ = vg.predict(W[idx]); z = (X[idx] - mp - mu_r) / sd_r; t2 = (z ** 2).sum(1)
                uc = np.unique(cc); Tc = np.array([t2[cc == kk].mean() for kk in uc])
                ic = Tc[uc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
                UCL = mu0 + K * sd0 * np.sqrt(LAM / (2 - LAM)); e = mu0; E = []
                for v in Tc: e = LAM * v + (1 - LAM) * e; E.append(e)
                cr = np.where(np.array(E) > UCL)[0]; trans = int(uc[cr[0]]) if len(cr) else int(uc[-1] + 1)
                normal_idx.append(idx[cc < trans])
            normal_idx = np.concatenate(normal_idx)
            rng = np.random.default_rng(sd)
            if len(normal_idx) > 4000: normal_idx = rng.choice(normal_idx, 4000, replace=False)
            gp = fit_mogp(W[normal_idx], X[normal_idx], m=18, steps=150)

            for resid in [False, True]:
                def make_res(store, Wd, Xd):
                    out = {}
                    for u, (idx, cc, dc) in store.items():
                        if raw: mk = dc >= thr; idx2, cc2 = idx[mk], cc[mk]
                        else: idx2, cc2 = idx, cc
                        if resid:
                            pr, dc2 = cond(gp, Wd[idx2], Xd[idx2]); rr = Xd[idx2] - pr
                            th2 = np.percentile(dc2, DROP * 100); km = dc2 >= th2
                        else:
                            pr = pmean(gp, Wd[idx2]); rr = Xd[idx2] - pr; km = np.ones(len(idx2), bool)
                        d = {s: {} for s in SENS}
                        for kk in np.unique(cc2):
                            sel = (cc2 == kk) & km
                            if sel.sum() == 0: sel = (cc2 == kk)
                            mv = rr[sel].mean(0)
                            for j, s in enumerate(SENS): d[s][int(kk)] = float(mv[j])
                        out[int(u)] = d
                    return out
                dev_res = make_res(dev_s, W, X); test_res = make_res(test_s, Wt, Xt)
                st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
                          np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in SENS}
                nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
                dHI, tHI, dl, tl = build_hi(nrm(dev_res), nrm(test_res), sd)
                rr = rul_rmse(dHI, tHI)
                name = ('raw+resid' if resid else 'raw') if raw else ('resid' if resid else 'none')
                results[name].append(rr)
                log(f'  seed{sd} {name:>9s}: RUL RMSE = {rr:.2f}')
                if raw and resid and not saved_plot:
                    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
                    for u, cy, d in dl: ax[0].plot(np.array(cy)/max(cy), dHI[u], lw=1.3, alpha=.8, label=str(u))
                    for u, cy, d in tl: ax[1].plot(np.array(cy)/max(cy), tHI[u], lw=1.5, alpha=.85, label=str(u))
                    for a, tt in zip(ax, ['DEV HI', 'TEST HI']):
                        a.axhline(1, color='r', ls='--', lw=1); a.axhline(.5, color='gray', ls=':', lw=1)
                        a.set_ylim(0, 1.15); a.set_xlabel('life fraction'); a.set_title(tt); a.grid(alpha=.3); a.legend(fontsize=7, ncol=2)
                    ax[0].set_ylabel('Health Index'); fig.suptitle('HI (raw+resid filter)'); fig.tight_layout()
                    fig.savefig(os.path.join(HERE, 'hi_curves_ablation.png'), dpi=140); plt.close(fig); saved_plot = True

    log('=' * 54)
    log(f'{"config":>10s} | {"mean RUL RMSE":>13s} | {"std":>6s} | runs')
    log('-' * 54)
    for k in ['none', 'raw', 'resid', 'raw+resid']:
        v = np.array(results[k]); log(f'{k:>10s} | {v.mean():13.2f} | {v.std():6.2f} | {len(v)}')
    log('=' * 54)
    log('reference (old ExactGP pipeline) = 9.67')


if __name__ == '__main__':
    main()
