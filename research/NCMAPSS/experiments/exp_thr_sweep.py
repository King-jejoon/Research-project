"""
detcov-filter THRESHOLD sweep (FAST).  Sweeps how much high-uncertainty data to
drop at the residual stage.  Per seed the expensive parts (detection, Stage-2 GP,
per-point residual + detcov) are computed ONCE; then only the DROP fraction varies
(re-threshold -> re-aggregate per cycle -> retrain HI -> RUL).

DROP in {0.00, 0.05, 0.10, 0.15, 0.25, 0.40}   (0.00 = no filter)
Config: CPU, float32, 6 threads.
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
from vecchia_gp import VecchiaGP
from gpytorch_mogp_vecchia import GPyTorchMOGP, train_vecchia_mogp
from demo_cond import conditional_stats
import neural_fusion as NF

DT = torch.float32
SENS = ['T48', 'T50', 'Wf']; SIDX = [L.OUTPUT_NAMES.index(s) for s in SENS]
WARMUP, K, LAM, NPER = 10, 5.0, 0.2, 20
DROPS = [0.00, 0.05, 0.10, 0.15, 0.25, 0.40]
SEEDS = [0, 1, 2]
RES = os.path.join(HERE, 'thr_sweep_results.txt')


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


def cond(gp, Wq, Yq):
    teX = torch.tensor(gp['xs'].transform(Wq), dtype=DT)
    teY = torch.tensor(gp['ys'].transform(Yq), dtype=DT)
    pred, detcov, _ = conditional_stats(gp['model'], gp['lik'], gp['tX'], gp['tY'], gp['struct'], teX, m=18, test_y=teY)
    return pred * gp['ys'].scale_ + gp['ys'].mean_, detcov


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
    P, T = [], []
    for f in [0.5, 0.6, 0.7, 0.8]:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n*f)); P.append(rul(h[:cut])); T.append(n - cut)
    return float(np.sqrt(((np.array(P)-np.array(T))**2).mean()))


def main():
    open(RES, 'w').close()
    log(f'THRESHOLD SWEEP (fast)  drops={DROPS}  seeds={SEEDS}  NPER={NPER}')
    cache = L.load_cache(); t0 = time.time()
    W, Xall, A = cache['W_dev'], cache['X_s_dev'], cache['A_dev']
    Wt, Xtall, At = cache['W_test'], cache['X_s_test'], cache['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))
    results = {d: [] for d in DROPS}

    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        # --- detection (no filter) -> Stage 2 (once) ---
        normal_idx = []
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
            normal_idx.append(idx[cc < trans])
        normal_idx = np.concatenate(normal_idx)
        rng = np.random.default_rng(sd)
        if len(normal_idx) > 4000: normal_idx = rng.choice(normal_idx, 4000, replace=False)
        gp = fit_mogp(W[normal_idx], X[normal_idx], m=18, steps=150)

        # --- precompute residual + detcov ONCE per unit ---
        def collect(Wd, Xd, Ad, units):
            store = {}
            for u in units:
                rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
                ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 5 + sd)
                idx = np.concatenate(sel); cc = np.concatenate([np.full(len(g), k) for k, g in zip(ucyc, sel)])
                pr, dc = cond(gp, Wd[idx], Xd[idx])
                store[int(u)] = (cc, Xd[idx] - pr, dc)
            return store
        dev_store = collect(W, X, A, du); test_store = collect(Wt, Xt, At, tu)
        all_dc = np.concatenate([dev_store[u][2] for u in dev_store] + [test_store[u][2] for u in test_store])
        log(f'  seed{sd}: precompute done ({time.time()-t0:.0f}s), sweeping drops ...')

        # --- sweep drop (cheap) ---
        for drop in DROPS:
            thr = -np.inf if drop == 0 else np.percentile(all_dc, drop * 100)
            def agg(store):
                out = {}
                for u, (cc, resid, dc) in store.items():
                    keep = dc >= thr
                    d = {s: {} for s in SENS}
                    for kk in np.unique(cc):
                        sel = (cc == kk) & keep
                        if sel.sum() == 0: sel = (cc == kk)
                        mv = resid[sel].mean(0)
                        for j, s in enumerate(SENS): d[s][int(kk)] = float(mv[j])
                    out[u] = d
                return out
            dev_res, test_res = agg(dev_store), agg(test_store)
            st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
                      np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in SENS}
            nrm = lambda dd: {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
            def blist(dn):
                out = []
                for u in sorted(dn):
                    cy = sorted(dn[u][SENS[0]].keys()); out.append((u, np.array([[dn[u][s][k] for s in SENS] for k in cy])))
                return out
            dl, tl = blist(nrm(dev_res)), blist(nrm(test_res))
            np.random.seed(sd)
            him, _ = NF.train_model([d for _, d in dl], epochs=1000, lambda0=1.0, lambda1=6.0, lambda2=2.0,
                                    init_threshold=0.2, alpha=0.001, verbose=False)
            dHI = {u: him.forward(d).flatten() for u, d in dl}; tHI = {u: him.forward(d).flatten() for u, d in tl}
            rr = rul_of(dHI, tHI); results[drop].append(rr)
            log(f'  seed{sd} drop={drop:.2f} (kept {100*(1-drop):.0f}%): RUL RMSE = {rr:.2f}')

    log('=' * 52)
    log(f'{"drop":>6s} | {"kept":>5s} | {"mean RMSE":>9s} | {"std":>5s}')
    log('-' * 52)
    for d in DROPS:
        v = np.array(results[d]); log(f'{d:6.2f} | {100*(1-d):4.0f}% | {v.mean():9.2f} | {v.std():5.2f}')
    log('=' * 52)
    log(f'wall time = {time.time()-t0:.0f}s | reference old pipeline = 9.67')


if __name__ == '__main__':
    main()
