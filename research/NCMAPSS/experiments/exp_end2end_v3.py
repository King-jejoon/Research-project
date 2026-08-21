"""
END-TO-END v3
  Stage 0 (NEW): RAW-DATA uncertainty filter (detcov) BEFORE anything else.
                 A reference MOGP-Vecchia (dev run-in) scores every sampled point;
                 points with high predictive uncertainty (extreme operating params)
                 are dropped globally.  All later stages use the cleaned data.
  Stage 1: per-unit T^2 + EWMA detection on cleaned data -> predicted-normal
  Stage 2: MOGP-Vecchia on predicted-normal
  Stage 3: residuals r = X - Xhat  (per-cycle mean)
  Stage 4: neural HI  --> ALSO SAVES hi_curves_v3.png  (dev + test HI curves)
  Stage 5: Bayesian truncation-RUL RMSE

detcov is batched PER UNIT (one conditional_stats call/unit) to avoid the
per-cycle overhead that made v2b take hours.
Config: CPU, float32, 6 threads.
"""
import os, sys, time
import numpy as np
import torch
torch.set_num_threads(6)
import gpytorch
import matplotlib
matplotlib.use('Agg')
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
WARMUP, K, LAM, NPER = 10, 5.0, 0.2, 20
DETCOV_DROP = 0.15          # drop this fraction of highest-uncertainty (extreme-W) points
# neural HI physics-informed loss weights (TUNE these for HI shape)
L0, L1, L2, INIT_THR, EPOCHS = 1.0, 6.0, 2.0, 0.2, 1000


def log(s): print(s, flush=True)


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


def detcov_of(gp0, Wq, Yq):
    teX = torch.tensor(gp0['xs'].transform(Wq), dtype=DT)
    teY = torch.tensor(gp0['ys'].transform(Yq), dtype=DT)
    _, detcov, _ = conditional_stats(gp0['model'], gp0['lik'], gp0['tX'], gp0['tY'],
                                     gp0['struct'], teX, m=18, test_y=teY)
    return detcov


def predict_mean(gp, Wq):
    q = torch.tensor(gp['xs'].transform(Wq), dtype=DT)
    with torch.no_grad():
        mu = vecchia_mogp_predictive_mean(gp['model'], gp['lik'], gp['tX'], gp['tY'], q, gp['struct'], m=18)
    return mu.cpu().numpy() * gp['ys'].scale_ + gp['ys'].mean_


def main():
    t0 = time.time()
    c = L.load_cache()
    W, Xall, A = c['W_dev'], c['X_s_dev'], c['A_dev']
    Wt, Xtall, At = c['W_test'], c['X_s_test'], c['A_test']
    X, Xt = Xall[:, SIDX], Xtall[:, SIDX]
    du = np.unique(A[:, 0].astype(int)); tu = np.unique(At[:, 0].astype(int))

    # ---------- reference model for the RAW uncertainty filter ----------
    log('[ref] fitting reference MOGP-Vecchia on dev run-in ...')
    seed = []
    for u in du:
        rows = np.where(A[:, 0].astype(int) == u)[0]; cyc = A[rows, 1].astype(int)
        s = rows[cyc <= WARMUP]; rng = np.random.default_rng(int(u))
        if len(s) > 160: s = rng.choice(s, 160, replace=False)
        seed.append(s)
    seed = np.concatenate(seed)
    gp0 = fit_mogp(W[seed], X[seed], m=18, steps=120)

    # ---------- Stage 0: sample + detcov for every unit (batched) ----------
    log('[Stage 0] raw detcov (uncertainty) scoring ...')
    def sample_unit(Wd, Xd, Ad, u):
        rows = np.where(Ad[:, 0].astype(int) == u)[0]; cyc = Ad[rows, 1].astype(int)
        ucyc, sel = groups_of(rows, cyc, NPER, seed=int(u) * 7 + 1)
        idx = np.concatenate(sel); ccarr = np.concatenate([np.full(len(g), cc) for cc, g in zip(ucyc, sel)])
        dc = detcov_of(gp0, Wd[idx], Xd[idx])
        return idx, ccarr, dc
    dev_s = {int(u): sample_unit(W, X, A, u) for u in du}
    test_s = {int(u): sample_unit(Wt, Xt, At, u) for u in tu}
    all_dc = np.concatenate([dev_s[u][2] for u in dev_s] + [test_s[u][2] for u in test_s])
    thr = np.percentile(all_dc, DETCOV_DROP * 100)
    kept = (all_dc >= thr).mean()
    log(f'[Stage 0] detcov threshold={thr:.3f}  kept {kept*100:.0f}% of points (dropped extreme-W {100-kept*100:.0f}%)')

    # ---------- Stage 1: per-unit T^2 detection on CLEANED data ----------
    log('[Stage 1] detection on cleaned data ...')
    normal_idx = []
    for u in du:
        idx, cc, dc = dev_s[int(u)]; keep = dc >= thr
        idx, cc = idx[keep], cc[keep]
        # per-unit run-in normal model for T^2
        rin = idx[cc <= WARMUP]
        vg = VecchiaGP(m=15).fit(W[rin], X[rin], iters=100, verbose=False)
        ms, _ = vg.predict(W[rin]); r0 = X[rin] - ms; mu_r, sd_r = r0.mean(0), r0.std(0) + 1e-8
        mp, _ = vg.predict(W[idx]); z = (X[idx] - mp - mu_r) / sd_r; t2 = (z ** 2).sum(1)
        ucyc = np.unique(cc); Tc = np.array([t2[cc == k].mean() for k in ucyc])
        ic = Tc[ucyc <= WARMUP]; mu0, sd0 = ic.mean(), ic.std() + 1e-8
        UCL = mu0 + K * sd0 * np.sqrt(LAM / (2 - LAM))
        e = mu0; E = []
        for v in Tc: e = LAM * v + (1 - LAM) * e; E.append(e)
        cross = np.where(np.array(E) > UCL)[0]
        trans = int(ucyc[cross[0]]) if len(cross) else int(ucyc[-1] + 1)
        normal_idx.append(idx[cc < trans])
        log(f'   unit {u}: transition = {trans}')
    normal_idx = np.concatenate(normal_idx)
    rng = np.random.default_rng(0)
    if len(normal_idx) > 5000: normal_idx = rng.choice(normal_idx, 5000, replace=False)
    log(f'[Stage 1] predicted-normal pool = {len(normal_idx)}')

    # ---------- Stage 2 ----------
    log('[Stage 2] MOGP-Vecchia on predicted-normal ...')
    gp = fit_mogp(W[normal_idx], X[normal_idx], m=18, steps=150)

    # ---------- Stage 3: residuals on CLEANED data ----------
    log('[Stage 3] residuals (cleaned) ...')
    def resid(store, Wd, Xd):
        out = {}
        for u, (idx, cc, dc) in store.items():
            keep = dc >= thr; idx, cc = idx[keep], cc[keep]
            pm = predict_mean(gp, Wd[idx]); r = Xd[idx] - pm
            d = {s: {} for s in SENS}
            for k in np.unique(cc):
                mk = r[cc == k].mean(0)
                for j, s in enumerate(SENS): d[s][int(k)] = float(mk[j])
            out[int(u)] = d
        return out
    dev_res = resid(dev_s, W, X); test_res = resid(test_s, Wt, Xt)
    st = {s: (np.mean([v for u in dev_res for v in dev_res[u][s].values()]),
              np.std([v for u in dev_res for v in dev_res[u][s].values()]) + 1e-8) for s in SENS}
    def norm(dd): return {u: {s: {k: (v - st[s][0]) / st[s][1] for k, v in dd[u][s].items()} for s in SENS} for u in dd}
    dev_n, test_n = norm(dev_res), norm(test_res)

    # ---------- Stage 4: neural HI + PLOT ----------
    log('[Stage 4] neural HI ...')
    def blist(dn):
        out = []
        for u in sorted(dn):
            cy = sorted(dn[u][SENS[0]].keys())
            out.append((u, cy, np.array([[dn[u][s][k] for s in SENS] for k in cy])))
        return out
    dev_list, test_list = blist(dev_n), blist(test_n)
    him, loss_hist = NF.train_model([d for _, _, d in dev_list], epochs=EPOCHS, lambda0=L0,
                                    lambda1=L1, lambda2=L2, init_threshold=INIT_THR, alpha=0.001, verbose=False)
    devHI = {u: (np.array(cy), him.forward(d).flatten()) for u, cy, d in dev_list}
    testHI = {u: (np.array(cy), him.forward(d).flatten()) for u, cy, d in test_list}

    # ---- HI curve plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for u, (cy, h) in devHI.items():
        ax[0].plot(cy / cy.max(), h, lw=1.4, alpha=0.8, label=f'{u}')
    for u, (cy, h) in testHI.items():
        ax[1].plot(cy / cy.max(), h, lw=1.6, alpha=0.85, label=f'{u}')
    for a, t in zip(ax, ['DEV HI', 'TEST HI']):
        a.axhline(1.0, color='r', ls='--', lw=1); a.axhline(0.5, color='gray', ls=':', lw=1)
        a.set_ylim(0, 1.15); a.set_xlabel('life fraction'); a.set_title(t); a.grid(alpha=0.3); a.legend(fontsize=7, ncol=2)
    ax[0].set_ylabel('Health Index')
    fig.suptitle(f'HI curves  (loss weights lambda0={L0}, lambda1={L1}, lambda2={L2}, init_thr={INIT_THR})')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'hi_curves_v3.png'), dpi=140); plt.close(fig)
    # HI shape diagnostics
    starts = [h[0] for _, h in devHI.values()] + [h[0] for _, h in testHI.values()]
    mono = np.mean([np.mean(np.diff(h) >= -1e-6) for _, h in list(devHI.values()) + list(testHI.values())])
    log(f'[Stage 4] HI start: min={min(starts):.2f} max={max(starts):.2f} (want <0.5) | '
        f'monotonic frac={mono:.2f} | saved hi_curves_v3.png')

    # ---------- Stage 5: RUL ----------
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
    log('=' * 62)
    log('v3  (raw detcov filter + T^2 detect + MOGP-Vecchia)')
    log(f'  truncation RUL RMSE = {rmse:.2f} cycles   (mean true RUL {T.mean():.1f})')
    log(f'  v1 = 8.18 | reference (old HI) = 9.67')
    log(f'  wall time = {time.time()-t0:.0f}s')
    log('=' * 62)


if __name__ == '__main__':
    main()
