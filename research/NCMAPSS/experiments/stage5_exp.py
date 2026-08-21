"""stage5_exp.py — OFFICIAL Stage 5 (2026-07-17 final revision, user decision).

Bayesian regression basis: EXPONENTIAL degradation model
    h(c) = g0 + g2 (e^{beta c} - 1),   c = cycle / 500
- start is the curve minimum by construction (monotone increasing for g2 > 0;
  empirically g2 < 0 occurred in 0 of all dev/test fits on DS03 & DS01)
- beta is selected WITHOUT test data: leave-one-unit-out RUL cross-validation
  on the dev HI curves (NASA-score criterion), grid BETA_GRID.
Replaces the interim [1, c^2] basis (stage5_mono.py) and the old non-physical
[1, c, c^2] basis (exp_ds03_port.rul_eval).
"""
import numpy as np
from numpy.linalg import inv
from scipy import stats as sstats

C = (np.arange(500) / 500.0)
BETA_GRID = [16, 20, 25, 30, 35, 40, 45, 50]
_FRACS_CV = [0.2, 0.4, 0.6, 0.8]


def _psi(beta):
    return np.column_stack([np.ones(500), np.exp(beta * C) - 1])


def _nasa(P, T):
    d = np.asarray(P, float) - np.asarray(T, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def _eval_basis(Psi, devHI, testHI, fracs):
    p_dim = Psi.shape[1]
    Yd = [devHI[u] for u in sorted(devHI)]
    gam = np.zeros((len(Yd), p_dim))
    for i, y in enumerate(Yd):
        gam[i] = np.linalg.lstsq(Psi[:len(y)], y, rcond=None)[0]
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gam[i])[3:]) ** 2).sum() /
                  max(len(Yd[i]) - p_dim - 1, 1) for i in range(len(Yd))])
    mu0, cov0 = gam.mean(0), np.cov(gam.T) + 1e-10 * np.eye(p_dim)

    def first_passage(mu, cov, ncut):
        fx = lambda pp: (pp@mu - 1) / max(np.sqrt(max(pp@cov@pp.T, 1e-12)), 1e-9)
        tmin = ncut - 1; pmin = 0
        for tt in range(ncut - 1, 500):
            pr = sstats.norm.cdf(fx(Psi[tt]))
            if pr > pmin and pr <= 0.5: pmin = pr; tmin = tt
            if pr > 0.9: break
        tmax = min(tmin + 1, 499); pmax = sstats.norm.cdf(fx(Psi[tmax]))
        return (tmax if pmax == pmin else tmax - (tmax-tmin)*(pmax-0.5)/(pmax-pmin)) - ncut + 1

    P, T, FR, UN = [], [], [], []
    for f in fracs:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n * f))
            pm = Psi[:cut]
            An = inv((pm.T@pm)/s2 + inv(cov0))
            mu = An @ ((pm.T@h[:cut])/s2 + inv(cov0)@mu0)
            P.append(first_passage(mu, An, cut)); T.append(n - cut); FR.append(f); UN.append(u)
    return np.array(P), np.array(T), np.array(FR), np.array(UN)


def select_beta(devHI, grid=BETA_GRID):
    """leave-one-dev-unit-out truncation-RUL CV, NASA-score criterion."""
    us = sorted(devHI)
    best = None
    for b in grid:
        Psi = _psi(b); tot = 0.0
        for u in us:
            dtr = {v: devHI[v] for v in us if v != u}
            P, T, _, _ = _eval_basis(Psi, dtr, {u: devHI[u]}, _FRACS_CV)
            tot += _nasa(P, T)
        if best is None or tot < best[1]: best = (b, tot)
    return best[0]


def rul_eval_exp(devHI, testHI, fracs, beta=None):
    """official Stage 5. beta=None -> dev-CV selection (no test data used)."""
    if beta is None:
        beta = select_beta(devHI)
    P, T, FR, UN = _eval_basis(_psi(beta), devHI, testHI, fracs)
    return P, T, FR, UN, beta
