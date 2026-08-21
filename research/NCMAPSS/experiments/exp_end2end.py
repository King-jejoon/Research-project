"""
END-TO-END pipeline with the Stage-1 detector CONNECTED to Stage-2:

  Stage 1  per-dev-unit monitor -> predicted normal mask (no hs labels)
  Stage 2  MOGP-Vecchia (colleague code) trained on the PREDICTED-normal points
  Stage 3  per-cycle mean residual r = X - Xhat  for {T48,T50,Wf}, all dev+test cycles
  Stage 4  neural_fusion (existing MLP + physics-informed loss) -> HI
  Stage 5  Bayesian first-passage RUL, evaluated by TRUNCATION (RUL_test.txt is
           just the cycle counts, so a proper truncation-based RUL RMSE is used).

Config: CPU, float32, 6 threads.  Writes new dev/test HI and prints RUL RMSE.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
from scipy import stats
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
import exp_lib as L
from vecchia_gp import VecchiaGP
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp, vecchia_mogp_predictive_mean
import neural_fusion as NF

DEV, DT = 'cpu', torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, K, LAM = 10, 5.0, 0.2
NPER = 30


def log(s): print(s, flush=True)


def per_cycle_groups(rows, cyc, n_per, seed):
    rng = np.random.default_rng(seed); out = []
    for cc in np.unique(cyc):
        r = rows[cyc == cc]
        if len(r) > n_per: r = rng.choice(r, n_per, replace=False)
        out.append(r)
    return np.unique(cyc), out


def detect_transition(vg, mu_r, sd_r, W, X, rows, cyc, seed):
    ucyc, sel = per_cycle_groups(rows, cyc, NPER, seed)
    allidx = np.concatenate(sel)
    mean_p, _ = vg.predict(W[allidx]); z = (X[allidx] - mean_p - mu_r) / sd_r
    t2 = (z ** 2).sum(1); off = 0; Tc = []
    for r in sel: n = len(r); Tc.append(t2[off:off+n].mean()); off += n
    Tc = np.array(Tc)
    ic = Tc[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
    UCL = mu0 + K * sd0 * np.sqrt(LAM / (2 - LAM))
    e = mu0; E = []
    for t in Tc: e = LAM * t + (1 - LAM) * e; E.append(e)
    cross = np.where(np.array(E) > UCL)[0]
    return int(ucyc[cross[0]]) if len(cross) else int(ucyc[-1] + 1)


def main():
    t0 = time.time()
    c = L.load_cache()
    W, Xall, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    Wt, Xtall, At = c['W_test'], c['X_s_test'], c['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))

    # ---------- Stage 1: predicted-normal mask per dev unit ----------
    log('[Stage 1] detecting normal/abnormal per dev unit (no hs labels) ...')
    normal_idx = []
    for u in du:
        rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
        seed = rows[cyc <= WARMUP]
        rng = np.random.default_rng(int(u) * 13)
        if len(seed) > 1200: seed = rng.choice(seed, 1200, replace=False)
        vg = VecchiaGP(m=15).fit(W[seed], X[seed], iters=120, verbose=False)
        ms, _ = vg.predict(W[seed]); res = X[seed] - ms
        mu_r, sd_r = res.mean(0), res.std(0) + 1e-8
        trans = detect_transition(vg, mu_r, sd_r, W, X, rows, cyc, seed=int(u)*91)
        pn = rows[cyc < trans]                      # predicted-normal points
        normal_idx.append(pn)
        log(f'   unit {u}: predicted transition cycle = {trans}  (normal pts {len(pn)})')
    normal_idx = np.concatenate(normal_idx)
    rng = np.random.default_rng(0)
    if len(normal_idx) > 5000: normal_idx = rng.choice(normal_idx, 5000, replace=False)
    log(f'[Stage 1] predicted-normal training pool = {len(normal_idx)} pts')

    # ---------- Stage 2: MOGP-Vecchia on predicted-normal ----------
    log('[Stage 2] training MOGP-Vecchia on predicted-normal ...')
    xs = StandardScaler().fit(W[normal_idx]); ys = StandardScaler().fit(X[normal_idx])
    tX = torch.tensor(xs.transform(W[normal_idx]), dtype=DT)
    tY = torch.tensor(ys.transform(X[normal_idx]), dtype=DT)
    model = GPyTorchMOGP(4, num_tasks=3, rank=2, kernel='rbf').to(DEV, DT)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3).to(DEV, DT)
    res = train_vecchia_mogp(model, lik, tX, tY, m=20, num_steps=150, lr=0.05, group=True, verbose=False)
    model.eval(); lik.eval()

    def predict(Wq):
        q = torch.tensor(xs.transform(Wq), dtype=DT)
        with torch.no_grad():
            mu = vecchia_mogp_predictive_mean(model, lik, tX, tY, q, res.structure, m=20)
        return mu.cpu().numpy() * ys.scale_ + ys.mean_

    # ---------- Stage 3: per-cycle mean residual, all cycles ----------
    log('[Stage 3] residuals for all dev+test cycles ...')
    def resid_by_unit(Wd, Xd, Ad, units):
        out = {}
        for u in units:
            rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
            ucyc, sel = per_cycle_groups(rows, cyc, 40, seed=int(u)*5+1)
            allidx = np.concatenate(sel); pm = predict(Wd[allidx]); rr = Xd[allidx] - pm
            off = 0; d = {s: {} for s in SENS}
            for cc, r in zip(ucyc, sel):
                n = len(r); m = rr[off:off+n].mean(0); off += n
                for j, s in enumerate(SENS): d[s][int(cc)] = float(m[j])
            out[int(u)] = d
        return out
    dev_res = resid_by_unit(W, X, A, du)
    test_res = resid_by_unit(Wt, Xt, At, tu)

    # normalize (dev stats)
    stats_ = {}
    for s in SENS:
        vals = [v for u in dev_res for v in dev_res[u][s].values()]
        stats_[s] = (np.mean(vals), np.std(vals) + 1e-8)
    def norm(d):
        return {u: {s: {cc: (v - stats_[s][0]) / stats_[s][1] for cc, v in d[u][s].items()} for s in SENS} for u in d}
    dev_n, test_n = norm(dev_res), norm(test_res)

    # ---------- Stage 4: neural HI ----------
    log('[Stage 4] training neural HI ...')
    def build_list(dn):
        lst = []
        for u in sorted(dn):
            cycles = sorted(dn[u][SENS[0]].keys())
            lst.append((u, cycles, np.array([[dn[u][s][cc] for s in SENS] for cc in cycles])))
        return lst
    dev_list = build_list(dev_n); test_list = build_list(test_n)
    himodel, _ = NF.train_model([d for _, _, d in dev_list], epochs=1000,
                                lambda0=1.0, lambda1=6.0, lambda2=2.0, init_threshold=0.2,
                                alpha=0.001, verbose=False)
    def hi_of(lst):
        return {u: (np.array(cyc), himodel.forward(d).flatten()) for u, cyc, d in lst}
    devHI, testHI = hi_of(dev_list), hi_of(test_list)

    # ---------- Stage 5: truncation RUL RMSE ----------
    log('[Stage 5] RUL ...')
    ntr = len(dev_list)
    t = np.arange(500).reshape(-1, 1) / 500; Psi = np.hstack((np.ones((500, 1)), t, t**2))
    Yd = [devHI[u][1] for u in sorted(devHI)]
    gamma = np.zeros((ntr, 3))
    for i, y in enumerate(Yd):
        n = len(y); Xp = Psi[:n]; gamma[i] = (inv(Xp.T@Xp)@Xp.T@y).T
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gamma[i])[3:]) ** 2).sum() /
                  (len(Yd[i]) - 4) for i in range(ntr)])
    mu0, cov0 = gamma.mean(0), np.cov(gamma.T)
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
    P, T = np.array(P), np.array(T)
    rmse = float(np.sqrt(((P - T) ** 2).mean()))
    log('=' * 56)
    log(f'CONNECTED pipeline (predicted-normal + MOGP-Vecchia)')
    log(f'  truncation RUL RMSE = {rmse:.2f} cycles  (mean true RUL {T.mean():.1f})')
    log(f'  reference (existing HI)         = 9.67 cycles')
    log(f'  total wall time = {time.time()-t0:.0f}s')
    log('=' * 56)
    with open(os.path.join(HERE, 'end2end_results.txt'), 'w') as fh:
        fh.write(f'CONNECTED (predicted-normal + MOGP-Vecchia) RUL RMSE = {rmse:.2f} cycles\n')
        fh.write(f'reference existing-HI RUL RMSE = 9.67 cycles\n')
        for f, i in [(fr, i) for fr in [0.5,0.6,0.7,0.8] for i in range(len(testHI))]:
            pass


if __name__ == '__main__':
    main()
