"""
exp_stage5_zoo.py — Stage 5 model-family comparison. HI is FIXED at the
official configuration (lambda=(1,1,2,0.25)+L_flat(300,.002), causal drop 85%);
only the Bayesian regression basis changes. All candidates (except the
non-physical reference) satisfy start-is-minimum: no decrease-then-increase.

  quad      h = g0 + g2 c^2                      (current revised basis)
  cubic     h = g0 + g2 c^2 + g3 c^3
  quartic   h = g0 + g2 c^2 + g4 c^4
  exp(b)    h = g0 + g2 (e^{bc} - 1)             b chosen by dev SSE per seed
  cpquad    h = g0 + g2 max(c - tau, 0)^2        tau by marginal likelihood
  ref-full  h = g0 + g1 c + g2 c^2               (old, non-physical reference)

Eval: DS03 + DS01, 3 seeds, truncation 20/40/60/80%, RMSE + NASA score.
Run: /opt/anaconda3/envs/pt_prac/bin/python3 exp_stage5_zoo.py
"""
import os, sys, time
import numpy as np
from numpy.linalg import inv, slogdet
from scipy import stats as sstats

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, __import__('exp_paths').DEMO)
from exp_gp_selfdrop import build_resid
from exp_ds03_port import nasa_score, FRACS
from neural_fusion_tail import train_model_tail

SEEDS = [0, 1, 2]
DROP = 0.85
HI_CFG = dict(epochs=1000, lambda0=1.0, lambda1=2.0, lambda2=0.25,
              init_threshold=0.2, flat_w=300.0, flat_m=0.002, alpha=0.001)
CACHES = {'DS03': 'port_stats_s3_matern32_r1_m18_n8192_p200_s{sd}.npz',
          'DS01': 'port_stats_ds01_s3_matern32_r1_m18_n8192_p200_s{sd}.npz'}
RES = os.path.join(HERE, 'stage5_zoo_results.txt')
C = (np.arange(500) / 500.0)


def log(s):
    print(s, flush=True)
    with open(RES, 'a') as f: f.write(s + '\n')


def first_passage(Psi, mu, cov, ncut):
    fx = lambda pp: (pp@mu - 1) / max(np.sqrt(max(pp@cov@pp.T, 1e-12)), 1e-9)
    tmin = ncut - 1; pmin = 0
    for tt in range(ncut - 1, 500):
        pr = sstats.norm.cdf(fx(Psi[tt]))
        if pr > pmin and pr <= 0.5: pmin = pr; tmin = tt
        if pr > 0.9: break
    tmax = min(tmin + 1, len(Psi) - 1); pmax = sstats.norm.cdf(fx(Psi[tmax]))
    return (tmax if pmax == pmin else tmax - (tmax-tmin)*(pmax-0.5)/(pmax-pmin)) - ncut + 1


def eval_linear_basis(Psi, devHI, testHI, fracs):
    """conjugate Bayesian linear model on a fixed feature matrix Psi (500 x p)."""
    p_dim = Psi.shape[1]
    Yd = [devHI[u] for u in sorted(devHI)]
    gam = np.zeros((len(Yd), p_dim))
    for i, y in enumerate(Yd):
        Xp = Psi[:len(y)]; gam[i] = np.linalg.lstsq(Xp, y, rcond=None)[0]
    dof = max(1, len(Yd[0]) - p_dim - 1)
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gam[i])[3:]) ** 2).sum() /
                  max(len(Yd[i]) - p_dim - 1, 1) for i in range(len(Yd))])
    mu0, cov0 = gam.mean(0), np.cov(gam.T) + 1e-10 * np.eye(p_dim)
    P, T, FR = [], [], []
    for f in fracs:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n * f))
            pm = Psi[:cut]
            An = inv((pm.T@pm)/s2 + inv(cov0))
            mu = An @ ((pm.T@h[:cut])/s2 + inv(cov0)@mu0)
            P.append(first_passage(Psi, mu, An, cut)); T.append(n - cut); FR.append(f)
    return np.array(P), np.array(T), np.array(FR)


def eval_cpquad(devHI, testHI, fracs, taus):
    """changepoint quadratic: per dev unit tau by SSE; per test prefix tau by
    (marginal likelihood x empirical tau prior)."""
    def feats(tau):
        z = np.maximum(C - tau, 0.0) ** 2
        return np.column_stack([np.ones(500), z])
    Yd = [devHI[u] for u in sorted(devHI)]
    dev_tau, dev_gam = [], []
    for y in Yd:
        best = None
        for tau in taus:
            Xp = feats(tau)[:len(y)]
            g = np.linalg.lstsq(Xp, y, rcond=None)[0]
            sse = ((y - Xp@g) ** 2).sum()
            if best is None or sse < best[0]: best = (sse, tau, g)
        dev_tau.append(best[1]); dev_gam.append(best[2])
    dev_tau = np.array(dev_tau); G = np.array(dev_gam)
    mu0, cov0 = G.mean(0), np.cov(G.T) + 1e-10 * np.eye(2)
    s2 = np.mean([((Yd[i] - feats(dev_tau[i])[:len(Yd[i])]@G[i]) ** 2).sum() /
                  max(len(Yd[i]) - 3, 1) for i in range(len(Yd))])
    t_mu, t_sd = dev_tau.mean(), dev_tau.std() + 1e-3
    logw = {tau: -0.5*((tau-t_mu)/t_sd)**2 for tau in taus}
    P, T, FR = [], [], []
    for f in fracs:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n * f))
            best = None
            for tau in taus:
                Phi = feats(tau)[:cut]
                S = s2*np.eye(cut) + Phi@cov0@Phi.T
                r = h[:cut] - Phi@mu0
                sign, ld = slogdet(S)
                ll = -0.5*(r @ np.linalg.solve(S, r)) - 0.5*ld + logw[tau]
                if best is None or ll > best[0]: best = (ll, tau)
            tau = best[1]; Psi = feats(tau)
            pm = Psi[:cut]
            An = inv((pm.T@pm)/s2 + inv(cov0))
            mu = An @ ((pm.T@h[:cut])/s2 + inv(cov0)@mu0)
            P.append(first_passage(Psi, mu, An, cut)); T.append(n - cut); FR.append(f)
    return np.array(P), np.array(T), np.array(FR)


def pick_exp_beta(devHI, betas):
    Yd = [devHI[u] for u in sorted(devHI)]
    best = None
    for b in betas:
        Psi = np.column_stack([np.ones(500), np.exp(b*C) - 1])
        sse = 0.0
        for y in Yd:
            Xp = Psi[:len(y)]; g = np.linalg.lstsq(Xp, y, rcond=None)[0]
            sse += ((y - Xp@g) ** 2).sum()
        if best is None or sse < best[0]: best = (sse, b)
    return best[1]


def main():
    open(RES, 'w').close()
    t0 = time.time()
    log('STAGE-5 MODEL ZOO — HI fixed (official), truncation 20/40/60/80%, 3 seeds')
    TAUS = [c/500 for c in [0, 5, 10, 15, 20, 25, 30, 35, 40]]
    BETAS = [2, 4, 6, 8, 12, 16, 20]
    for name, pat in CACHES.items():
        HIs = []
        for sd in SEEDS:
            z = np.load(os.path.join(HERE, pat.format(sd=sd)))
            dl, tl = build_resid(z, DROP)
            np.random.seed(sd)
            him, _ = train_model_tail([d for _, _, d in dl], **HI_CFG)
            HIs.append(({u: him.forward(d).flatten() for u, cy, d in dl},
                        {u: him.forward(d).flatten() for u, cy, d in tl}))
        log(f'\n===== {name} =====')
        models = [
            ('quad [1,c2]  (current)', lambda d, t: eval_linear_basis(
                np.column_stack([np.ones(500), C**2]), d, t, FRACS)),
            ('cubic [1,c2,c3]', lambda d, t: eval_linear_basis(
                np.column_stack([np.ones(500), C**2, C**3]), d, t, FRACS)),
            ('quartic [1,c2,c4]', lambda d, t: eval_linear_basis(
                np.column_stack([np.ones(500), C**2, C**4]), d, t, FRACS)),
            ('exp [1,e^bc-1]', None),   # handled below (needs beta per seed)
            ('cpquad max(c-tau,0)^2', lambda d, t: eval_cpquad(d, t, FRACS, TAUS)),
            ('REF full quad [1,c,c2]', lambda d, t: eval_linear_basis(
                np.column_stack([np.ones(500), C, C**2]), d, t, FRACS)),
        ]
        for mname, fn in models:
            rmses, scores = [], []
            for sd, (dHI, tHI) in zip(SEEDS, HIs):
                if mname.startswith('exp'):
                    b = pick_exp_beta(dHI, BETAS)
                    Psi = np.column_stack([np.ones(500), np.exp(b*C) - 1])
                    P, T, FR = eval_linear_basis(Psi, dHI, tHI, FRACS)
                else:
                    P, T, FR = fn(dHI, tHI)
                rmses.append(float(np.sqrt(((P - T) ** 2).mean())))
                scores.append(nasa_score(P, T))
            log(f'  {mname:>26}: RMSE {np.mean(rmses):6.2f} ±{np.std(rmses):5.2f}   '
                f'score {np.mean(scores):8.1f} ±{np.std(scores):6.1f}')
    log(f'\nwall {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
