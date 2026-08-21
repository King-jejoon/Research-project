"""stage5_mono.py — OFFICIAL Stage 5 (2026-07-17 revision).

Bayesian regression basis changed [1, c, c^2] -> [1, c^2] so that the fitted
health-index curve has its MINIMUM at the start (no decrease-then-increase,
which is physically impossible for damage). The linear term was found to be
negative in 100% of dev/test fits (vertex at cycle 11-17), i.e. the old basis
systematically produced an unphysical early dip; user decision: physical
validity takes precedence over the RMSE benefit of the dip.
"""
import numpy as np
from numpy.linalg import inv
from scipy import stats as sstats


def rul_eval_mono(devHI, testHI, fracs):
    """Bayesian first-passage truncation RUL with monotone basis [1, c^2]."""
    t = np.arange(500).reshape(-1, 1) / 500
    Psi = np.hstack((np.ones((500, 1)), t**2))
    Yd = [devHI[u] for u in sorted(devHI)]; ntr = len(Yd)
    gam = np.zeros((ntr, 2))
    for i, y in enumerate(Yd):
        Xp = Psi[:len(y)]; gam[i] = (inv(Xp.T@Xp)@Xp.T@y).T
    s2 = np.mean([((Yd[i][3:] - (Psi[:len(Yd[i])]@gam[i])[3:]) ** 2).sum() / (len(Yd[i]) - 3)
                  for i in range(ntr)])
    mu0, cov0 = gam.mean(0), np.cov(gam.T)

    def rul(HI):
        p = Psi[:len(HI)]
        mu = inv((p.T@p)/s2 + inv(cov0)) @ ((p.T@HI)/s2 + inv(cov0)@mu0)
        cov = inv((p.T@p)/s2 + inv(cov0))
        fx = lambda pp: (pp@mu - 1) / pow(pp@cov@pp.T, 0.5)
        tmin = len(HI) - 1; pmin = 0
        for tt in range(len(HI) - 1, 500):
            pr = sstats.norm.cdf(fx(Psi[tt]))
            if pr > pmin and pr <= 0.5: pmin = pr; tmin = tt
            if pr > 0.9: break
        tmax = min(tmin + 1, len(Psi) - 1); pmax = sstats.norm.cdf(fx(Psi[tmax]))
        return (tmax if pmax == pmin else tmax - (tmax-tmin)*(pmax-0.5)/(pmax-pmin)) - len(HI) + 1

    P, T, FR, UN = [], [], [], []
    for f in fracs:
        for u in sorted(testHI):
            h = testHI[u]; n = len(h); cut = max(4, int(n * f))
            P.append(rul(h[:cut])); T.append(n - cut); FR.append(f); UN.append(u)
    return np.array(P), np.array(T), np.array(FR), np.array(UN)
