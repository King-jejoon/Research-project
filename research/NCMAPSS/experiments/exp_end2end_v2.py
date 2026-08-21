"""
END-TO-END v2 — now includes the paper's 3 methods, using the DEMO code:

  #1 GP degradation modeling under varying operating conditions   -> MOGP-Vecchia (Stage 2)
  #2 detecting state change according to a CONDITION LIKELIHOOD    -> Stage 1 uses per-cycle
       conditional log-likelihood (LL) on an EWMA chart (replaces the T^2 statistic)
  #3 filtering out extreme-operating-parameter data by UNCERTAINTY -> Stage 3 drops points
       with low detcov (= high predictive uncertainty) before aggregating residuals

Stages 4-5 (neural HI, Bayesian truncation-RUL) unchanged.
Config: CPU, float32, 6 threads.  Compares to the reference RUL RMSE (existing HI = 9.67)
and to v1 (connected, no LL/filter = 8.18).
"""
import os, sys, time
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
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond import conditional_stats
import neural_fusion as NF

DT = torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, K, LAM, MCOND = 10, 4.0, 0.2, 20
NPER = 25
DETCOV_DROP = 0.20        # drop this fraction of highest-uncertainty points (#3)


def log(s): print(s, flush=True)


def groups_of(rows, cyc, n_per, seed):
    rng = np.random.default_rng(seed); out = []
    for cc in np.unique(cyc):
        r = rows[cyc == cc]
        if len(r) > n_per: r = rng.choice(r, n_per, replace=False)
        out.append(r)
    return np.unique(cyc), out


def fit_mogp(Wtr, Ytr, m=20, steps=150):
    xs = StandardScaler().fit(Wtr); ys = StandardScaler().fit(Ytr)
    tX = torch.tensor(xs.transform(Wtr), dtype=DT); tY = torch.tensor(ys.transform(Ytr), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=Ytr.shape[1], rank=2, kernel='rbf').to('cpu', DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=Ytr.shape[1]).to('cpu', DT)
    r = train_vecchia_mogp(model, lik, tX, tY, m=m, num_steps=steps, lr=0.05, group=True, verbose=False)
    model.eval(); lik.eval()
    return dict(model=model, lik=lik, tX=tX, tY=tY, struct=r.structure, xs=xs, ys=ys)


def cond_on(gp, Wq, Yq=None):
    teX = torch.tensor(gp['xs'].transform(Wq), dtype=DT)
    teY = None if Yq is None else torch.tensor(gp['ys'].transform(Yq), dtype=DT)
    pred, detcov, ll = conditional_stats(gp['model'], gp['lik'], gp['tX'], gp['tY'],
                                         gp['struct'], teX, m=MCOND, test_y=teY)
    pred = pred * gp['ys'].scale_ + gp['ys'].mean_        # de-standardize
    return pred, detcov, ll


def main():
    t0 = time.time()
    c = L.load_cache()
    W, Xall, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    Wt, Xtall, At = c['W_test'], c['X_s_test'], c['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))

    # ---------- Stage 1: CONDITION-LIKELIHOOD detection (#2) ----------
    log('[Stage 1] condition-likelihood detection ...')
    # PER-UNIT condition-likelihood detection: each unit gets its OWN normal model
    # fit on its own run-in, so cross-unit generalization can't corrupt detection.
    normal_idx = []
    for u in du:
        rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
        s = rows[cyc <= WARMUP]; rng = np.random.default_rng(int(u))
        if len(s) > 500: s = rng.choice(s, 500, replace=False)
        gpu = fit_mogp(W[s], X[s], m=15, steps=120)          # per-unit normal model
        ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7)
        llc = []
        for g in sel:
            _, _, ll = cond_on(gpu, W[g], X[g]); llc.append(np.mean(ll))
        llc = np.array(llc)
        ic = llc[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
        LCL = mu0 - K * sd0 * np.sqrt(LAM / (2 - LAM))       # abnormal -> LL drops
        e = mu0; E = []
        for v in llc: e = LAM * v + (1 - LAM) * e; E.append(e)
        cross = np.where(np.array(E) < LCL)[0]
        trans = int(ucyc[cross[0]]) if len(cross) else int(ucyc[-1] + 1)
        normal_idx.append(rows[cyc < trans])
        log(f'   unit {u}: LL-transition = {trans}')
    normal_idx = np.concatenate(normal_idx)
    rng = np.random.default_rng(0)
    if len(normal_idx) > 5000: normal_idx = rng.choice(normal_idx, 5000, replace=False)
    log(f'[Stage 1] predicted-normal pool = {len(normal_idx)}')

    # ---------- Stage 2: MOGP-Vecchia on predicted-normal (#1) ----------
    log('[Stage 2] MOGP-Vecchia on predicted-normal ...')
    gp = fit_mogp(W[normal_idx], X[normal_idx], m=20, steps=150)

    # ---------- Stage 3: residuals + UNCERTAINTY FILTER (#3) ----------
    log('[Stage 3] residuals with detcov uncertainty filter ...')
    def resid_by_unit(Wd, Xd, Ad, units):
        out = {}
        for u in units:
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = groups_of(rows, cyc, 30, seed=int(u) * 5 + 1)
            d = {s: {} for s in SENS}
            # per-cycle predictions + detcov, then a per-unit uncertainty threshold
            per = []
            for cc, g in zip(ucyc, sel):
                p, dc, _ = cond_on(gp, Wd[g], Xd[g])
                per.append((int(cc), p, dc, Xd[g]))
            dc_all = np.concatenate([e[2] for e in per])
            thr = np.percentile(dc_all, DETCOV_DROP * 100)     # keep detcov >= thr (low uncertainty)
            for cc, p, dc, tr in per:
                keep = dc >= thr
                if keep.sum() == 0: keep = np.ones(len(dc), bool)
                r = (tr[keep] - p[keep]).mean(0)
                for j, s in enumerate(SENS): d[s][cc] = float(r[j])
            out[int(u)] = d
        return out
    dev_res = resid_by_unit(W, X, A, du)
    test_res = resid_by_unit(Wt, Xt, At, tu)

    st = {}
    for s in SENS:
        vals = [v for u in dev_res for v in dev_res[u][s].values()]
        st[s] = (np.mean(vals), np.std(vals) + 1e-8)
    def norm(dd):
        return {u: {s: {cc: (v - st[s][0]) / st[s][1] for cc, v in dd[u][s].items()} for s in SENS} for u in dd}
    dev_n, test_n = norm(dev_res), norm(test_res)

    # ---------- Stage 4: neural HI ----------
    log('[Stage 4] neural HI ...')
    def blist(dn):
        out = []
        for u in sorted(dn):
            cyc = sorted(dn[u][SENS[0]].keys())
            out.append((u, cyc, np.array([[dn[u][s][cc] for s in SENS] for cc in cyc])))
        return out
    dev_list, test_list = blist(dev_n), blist(test_n)
    him, _ = NF.train_model([d for _, _, d in dev_list], epochs=1000, lambda0=1.0,
                            lambda1=6.0, lambda2=2.0, init_threshold=0.2, alpha=0.001, verbose=False)
    devHI = {u: (np.array(cy), him.forward(d).flatten()) for u, cy, d in dev_list}
    testHI = {u: (np.array(cy), him.forward(d).flatten()) for u, cy, d in test_list}

    # ---------- Stage 5: truncation RUL RMSE ----------
    log('[Stage 5] RUL ...')
    t = np.arange(500).reshape(-1, 1) / 500; Psi = np.hstack((np.ones((500, 1)), t, t**2))
    Yd = [devHI[u][1] for u in sorted(devHI)]; ntr = len(Yd)
    gam = np.zeros((ntr, 3))
    for i, y in enumerate(Yd):
        n = len(y); Xp = Psi[:n]; gam[i] = (inv(Xp.T@Xp)@Xp.T@y).T
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gam[i])[3:]) ** 2).sum() / (len(Yd[i]) - 4) for i in range(ntr)])
    mu0, cov0 = gam.mean(0), np.cov(gam.T)
    def rul(HI):
        p = Psi[:len(HI)]
        mu = inv((p.T@p)/s2 + inv(cov0)) @ ((p.T@HI)/s2 + inv(cov0)@mu0)
        cov = inv((p.T@p)/s2 + inv(cov0))
        fx = lambda pp: (pp@mu - 1) / pow(pp@cov@pp.T, 0.5)
        tmin = len(HI)-1; pmin = 0
        for tt in range(len(HI)-1, 500):
            pr = stats.norm.cdf(fx(Psi[tt]))
            if pr > pmin and pr <= 0.5: pmin = pr; tmin = tt
            if pr > 0.9: break
        tmax = tmin+1; pmax = stats.norm.cdf(fx(Psi[tmax]))
        tmid = tmax if pmax == pmin else tmax - (tmax-tmin)*(pmax-0.5)/(pmax-pmin)
        return tmid - len(HI) + 1
    P, T = [], []
    for f in [0.5, 0.6, 0.7, 0.8]:
        for u in sorted(testHI):
            h = testHI[u][1]; n = len(h); cut = max(4, int(n*f))
            P.append(rul(h[:cut])); T.append(n - cut)
    P, T = np.array(P), np.array(T); rmse = float(np.sqrt(((P-T)**2).mean()))
    log('=' * 60)
    log('v2  (predicted-normal + MOGP-Vecchia + LL-detect + detcov-filter)')
    log(f'  truncation RUL RMSE = {rmse:.2f} cycles   (mean true RUL {T.mean():.1f})')
    log(f'  v1 (no LL/filter)   = 8.18 cycles')
    log(f'  reference (old HI)  = 9.67 cycles')
    log(f'  wall time = {time.time()-t0:.0f}s')
    log('=' * 60)


if __name__ == '__main__':
    main()
