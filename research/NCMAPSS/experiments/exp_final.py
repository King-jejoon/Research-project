"""
FINAL pipeline (NO uncertainty filter) — the configuration the ablation showed
is the most stable.  Reported as mean +/- std over several seeds.

  Stage 1  per-unit T^2 + EWMA detection (no hs labels)      -> predicted-normal
  Stage 2  MOGP-Vecchia on predicted-normal                  -> baseline Xhat=f(W)
  Stage 3  per-cycle mean residual r = X - Xhat              (no filter)
  Stage 4  neural HI (MLP + physics-informed loss)           -> HI in [0,1]
  Stage 5  Bayesian first-passage, truncation RUL RMSE

Saves hi_curves_final.png (last seed) and writes final_results.txt.
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
import neural_fusion as NF

DT = torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, K, LAM, NPER = 10, 5.0, 0.2, 60           # NPER 60: smoother per-cycle residuals
L0, L1, L2, INIT_THR, EPOCHS = 1.0, 6.0, 2.0, 0.2, 1000
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'final_results.txt')


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


def rul_of(devHI, testHI):
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
    P, T = [], []
    for f in [0.5, 0.6, 0.7, 0.8]:
        pf, tf = [], []
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n*f)); pf.append(rul(h[:cut])); tf.append(n - cut)
        per_f[f] = float(np.sqrt(((np.array(pf)-np.array(tf))**2).mean())); P += pf; T += tf
    return float(np.sqrt(((np.array(P)-np.array(T))**2).mean())), per_f


def run_seed(sd, cache):
    torch.manual_seed(sd); np.random.seed(sd)
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    Wt, Xtall, At = cache['W_test'], cache['X_s_test'], cache['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))

    # Stage 1: detection
    normal_idx = []; trans_by_u = {}
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
        trans_by_u[int(u)] = trans; normal_idx.append(idx[cc < trans])
    normal_idx = np.concatenate(normal_idx)
    rng = np.random.default_rng(sd)
    if len(normal_idx) > 5000: normal_idx = rng.choice(normal_idx, 5000, replace=False)

    # Stage 2
    gp = fit_mogp(W[normal_idx], X[normal_idx], m=18, steps=150)

    # Stage 3: residuals (no filter)
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

    # Stage 4: HI
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
    rmse, per_f = rul_of(dHI, tHI)
    return rmse, per_f, trans_by_u, dl, tl, dHI, tHI


def main():
    open(RES, 'w').close()
    log(f'FINAL pipeline (NO filter)  seeds={SEEDS}  NPER={NPER}  lambda=({L0},{L1},{L2})')
    cache = L.load_cache(); t0 = time.time()
    rmses = []; last = None
    for sd in SEEDS:
        rmse, per_f, trans, dl, tl, dHI, tHI = run_seed(sd, cache)
        rmses.append(rmse)
        pf = ' '.join(f'{f:.0%}:{v:.2f}' for f, v in per_f.items())
        log(f'  seed{sd}: RUL RMSE = {rmse:.2f}  (by trunc {pf})  transitions={list(trans.values())}')
        last = (dl, tl, dHI, tHI)
    rmses = np.array(rmses)
    log('=' * 56)
    log(f'FINAL  RUL RMSE = {rmses.mean():.2f} +/- {rmses.std():.2f} cycles   (reference old pipeline = 9.67)')
    log(f'wall time = {time.time()-t0:.0f}s')
    log('=' * 56)

    # HI plot (last seed)
    dl, tl, dHI, tHI = last
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for u, cy, d in dl: ax[0].plot(np.array(cy)/max(cy), dHI[u], lw=1.3, alpha=.8, label=str(u))
    for u, cy, d in tl: ax[1].plot(np.array(cy)/max(cy), tHI[u], lw=1.5, alpha=.85, label=str(u))
    for a, tt in zip(ax, ['DEV HI', 'TEST HI']):
        a.axhline(1, color='r', ls='--', lw=1); a.axhline(.5, color='gray', ls=':', lw=1)
        a.set_ylim(0, 1.15); a.set_xlabel('life fraction'); a.set_title(tt); a.grid(alpha=.3); a.legend(fontsize=7, ncol=2)
    ax[0].set_ylabel('Health Index'); fig.suptitle('FINAL HI (no filter)'); fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'hi_curves_final.png'), dpi=140); plt.close(fig)
    log('saved hi_curves_final.png')


if __name__ == '__main__':
    main()
