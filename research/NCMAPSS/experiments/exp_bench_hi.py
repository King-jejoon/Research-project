"""exp_bench_hi.py — HI-construction benchmark library.

Benchmark slot (agreed design A):

    per-flight residual sequence (n_flights, 5)  ->  [ HI construction ]  ->  HI curve
    -> Bayesian exponential first-passage (SHARED, frozen)  ->  RUL

Input data is exactly what the frozen pipeline feeds to the HI network:
`{split}_{u}_trim` of rul_input_s{sd}.npz (gated trimmed-mean residual of
[T30,T48,T50,Nc,Wf], raw units), plus `_hours` as the flight-level covariate.

Methods (common interface: fit(R, cov, life) -> predict(r, c) -> HI):
  ours     constrained MLP 5-4-2-1, frozen HI_CFG        unsupervised (shape only)
  lr       OLS on normalised life                        supervised
  bspline  additive cubic B-spline on normalised life    supervised
  llke     local linear kernel estimation                supervised
  cabn     Wei et al. (2025) K=1 covariate-adjusted BN,
           HI = calibrated LR monitoring statistic       healthy sample only

Note on fairness: lr/bspline/llke consume the lifetime labels of the training
units, which `ours` never sees.  The baselines are therefore given strictly more
information; this is the conservative direction for our claim.
"""
import sys, os, time
import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import expm
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from neural_fusion_tail import train_model_tail

HI_CFG = dict(epochs=1000, lambda0=1.0, lambda1=8.0, lambda2=2.0,
              init_threshold=0.2, flat_w=800.0, flat_m=0.004, alpha=0.001)


# --------------------------------------------------------------------------- #
#  base                                                                        #
# --------------------------------------------------------------------------- #
class HIMethod:
    """R: dict u -> (n_u, 5) raw residual rows, ordered by cycle.
       cov: dict u -> (n_u, c) flight-level covariates (hours).
       Feature z-normalisation is fitted on the training units only."""
    name = 'base'
    supervised = False

    def __init__(self, **hp):
        self.hp = hp
        self.mu = self.sg = None

    # ---- helpers -----------------------------------------------------------
    def _znorm_fit(self, R, units):
        a = np.concatenate([R[u] for u in units])
        self.mu, self.sg = a.mean(0), a.std(0) + 1e-8

    def _z(self, r):
        return (r - self.mu) / self.sg

    @staticmethod
    def _life(n):
        """normalised life label: 0 at the first flight, 1 at failure."""
        return np.arange(n, dtype=float) / max(n - 1, 1)

    def _smooth(self, v):
        """optional TRAILING moving average (causal: never uses future flights).
        Offered to the baselines so that a noisy pointwise HI is not penalised
        for something a trivial post-filter would fix."""
        k = int(self.hp.get('smooth', 0))
        if k < 2:
            return v
        pad = np.r_[np.full(k - 1, v[0]), v]
        return np.convolve(pad, np.ones(k) / k, mode='valid')

    # ---- interface ---------------------------------------------------------
    def fit(self, R, cov, units, seed=0):
        raise NotImplementedError

    def predict(self, r, c):
        raise NotImplementedError

    def fit_all(self, R, cov, units, seed=0):
        """fit + return HI curves of the fitting units (timed)."""
        t0 = time.perf_counter()
        self.fit(R, cov, units, seed)
        self.t_fit = time.perf_counter() - t0
        return {u: self.predict(R[u], cov[u]) for u in units}


# --------------------------------------------------------------------------- #
#  1. ours — constrained MLP                                                   #
# --------------------------------------------------------------------------- #
class OursMLP(HIMethod):
    name = 'ours'

    def fit(self, R, cov, units, seed=0):
        self._znorm_fit(R, units)
        np.random.seed(seed)
        cfg = dict(HI_CFG); cfg.update({k: v for k, v in self.hp.items() if k in cfg})
        self.model, _ = train_model_tail([self._z(R[u]) for u in units], **cfg)

    def predict(self, r, c):
        return self.model.forward(self._z(r)).flatten()


# --------------------------------------------------------------------------- #
#  2. lr — linear regression on normalised life                                #
# --------------------------------------------------------------------------- #
class LinReg(HIMethod):
    name = 'lr'
    supervised = True

    def _design(self, z):
        return np.column_stack([np.ones(len(z)), z])

    def fit(self, R, cov, units, seed=0):
        self._znorm_fit(R, units)
        X = np.concatenate([self._design(self._z(R[u])) for u in units])
        y = np.concatenate([self._life(len(R[u])) for u in units])
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]

    def predict(self, r, c):
        return self._smooth(self._design(self._z(r)) @ self.beta)


# --------------------------------------------------------------------------- #
#  3. bspline — additive cubic B-spline on normalised life                     #
# --------------------------------------------------------------------------- #
class BSplineHI(HIMethod):
    name = 'bspline'
    supervised = True

    def __init__(self, n_knots=5, ridge=1e-6, **hp):
        super().__init__(n_knots=n_knots, ridge=ridge, **hp)
        self.n_knots = n_knots
        self.ridge = ridge

    def _basis(self, zj, j):
        t = self.knots[j]
        B = BSpline.design_matrix(np.clip(zj, t[3] + 1e-9, t[-4] - 1e-9), t, 3).toarray()
        return B[:, 1:]                       # drop one column, global intercept added

    def _design(self, z):
        return np.column_stack([np.ones(len(z))] +
                               [self._basis(z[:, j], j) for j in range(z.shape[1])])

    def fit(self, R, cov, units, seed=0):
        self._znorm_fit(R, units)
        Z = np.concatenate([self._z(R[u]) for u in units])
        q = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        self.knots = []
        for j in range(Z.shape[1]):
            lo, hi = Z[:, j].min(), Z[:, j].max()
            pad = 0.05 * (hi - lo) + 1e-6
            inner = np.quantile(Z[:, j], q)
            self.knots.append(np.r_[[lo - pad] * 4, inner, [hi + pad] * 4])
        X = self._design(Z)
        y = np.concatenate([self._life(len(R[u])) for u in units])
        A = X.T @ X + self.ridge * np.eye(X.shape[1])
        self.beta = np.linalg.solve(A, X.T @ y)

    def predict(self, r, c):
        return self._smooth(self._design(self._z(r)) @ self.beta)


# --------------------------------------------------------------------------- #
#  4. llke — local linear kernel estimation                                    #
# --------------------------------------------------------------------------- #
class LLKE(HIMethod):
    name = 'llke'
    supervised = True

    def __init__(self, h=0.75, ridge=1e-4, **hp):
        super().__init__(h=h, ridge=ridge, **hp)
        self.h = h
        self.ridge = ridge

    def fit(self, R, cov, units, seed=0):
        self._znorm_fit(R, units)
        self.Xtr = np.concatenate([self._z(R[u]) for u in units])
        self.ytr = np.concatenate([self._life(len(R[u])) for u in units])

    def predict(self, r, c):
        Zq = self._z(r)
        d = Zq[:, None, :] - self.Xtr[None, :, :]          # (nq, ntr, p)
        w = np.exp(-0.5 * (d / self.h) ** 2).prod(2)       # product Gaussian kernel
        out = np.empty(len(Zq))
        p = self.Xtr.shape[1]
        I = self.ridge * np.eye(p + 1)
        for i in range(len(Zq)):
            wi = w[i]
            if wi.sum() < 1e-12:                            # far from every design point
                out[i] = self.ytr[np.argmax(wi)]
                continue
            D = np.column_stack([np.ones(len(self.Xtr)), d[i]])
            A = D.T @ (wi[:, None] * D) + I
            b = D.T @ (wi * self.ytr)
            out[i] = np.linalg.solve(A, b)[0]               # local intercept = fit at r
        return self._smooth(out)


# --------------------------------------------------------------------------- #
#  5. cabn — Wei et al. (2025), K = 1 covariate-adjusted Bayesian network       #
# --------------------------------------------------------------------------- #
def _notears(R, lam1=0.05, max_iter=20, rho0=1.0, tol=1e-8):
    """min 0.5/n ||R - R W||_F^2 + lam1 ||W||_1  s.t.  h(W) = tr(e^{W o W}) - p = 0
    (Zheng et al. 2018 augmented Lagrangian; W has zero diagonal)."""
    n, p = R.shape
    S = R.T @ R / n
    rho, alpha, h_prev = rho0, 0.0, np.inf
    w = np.zeros(2 * p * p)

    def unpack(v):
        return (v[:p * p] - v[p * p:]).reshape(p, p)

    def obj(v):
        W = unpack(v)
        M = np.eye(p) - W
        loss = 0.5 * np.trace(M.T @ S @ M)
        G_loss = -S @ M
        E = expm(W * W)
        h = np.trace(E) - p
        G_h = E.T * W * 2
        f = loss + 0.5 * rho * h * h + alpha * h + lam1 * v.sum()
        G = G_loss + (rho * h + alpha) * G_h
        g = np.concatenate([G.ravel(), -G.ravel()]) + lam1
        return f, g

    bnds = []
    for k in range(2):
        for i in range(p):
            for j in range(p):
                bnds.append((0, 0) if i == j else (0, None))
    for _ in range(max_iter):
        while rho < 1e16:
            sol = minimize(obj, w, jac=True, method='L-BFGS-B', bounds=bnds)
            W = unpack(sol.x)
            h = np.trace(expm(W * W)) - p
            if h > 0.25 * h_prev:
                rho *= 10
            else:
                break
        w, h_prev = sol.x, h
        alpha += rho * h
        if h <= tol:
            break
    W = unpack(w)
    W[np.abs(W) < 0.05] = 0.0                 # standard NOTEARS thresholding
    return W


class CaBN(HIMethod):
    """K=1 CaBN:  X = beta + B^T z + R,  R = W^T R + eps,  eps ~ N(0, Sigma diag).
    Fitted on the in-control (healthy) flights of the training units only.
    HI = calibrated likelihood-ratio monitoring statistic
         T_t = r_t^T Phi^{-1} r_t,   Phi^{-1} = (I-W) Sigma^{-1} (I-W)^T."""
    name = 'cabn'

    def __init__(self, n_healthy=10, lam1=0.05, stat='sqrt', **hp):
        super().__init__(n_healthy=n_healthy, lam1=lam1, stat=stat, **hp)
        self.n_healthy = n_healthy
        self.lam1 = lam1
        self.stat = stat

    def _cov_design(self, c):
        return np.column_stack([np.ones(len(c)), c])       # [1, hours]

    def fit(self, R, cov, units, seed=0):
        self._znorm_fit(R, units)
        nh = self.n_healthy
        Xh = np.concatenate([self._z(R[u][:nh]) for u in units])
        Zh = np.concatenate([self._cov_design(cov[u][:nh]) for u in units])
        # covariate-adjusted mean  [beta; B]
        self.coef = np.linalg.lstsq(Zh, Xh, rcond=None)[0]
        Rh = Xh - Zh @ self.coef
        W = _notears(Rh, lam1=self.lam1)
        E = Rh - Rh @ W
        sig = E.var(0) + 1e-8
        M = np.eye(len(sig)) - W
        self.Pinv = M @ np.diag(1.0 / sig) @ M.T
        self.W, self.sigma = W, sig
        # dev calibration of the statistic onto the HI scale
        s_lo = np.mean([self._stat(R[u][:nh], cov[u][:nh]).mean() for u in units])
        s_hi = np.mean([self._stat(R[u][-3:], cov[u][-3:]).mean() for u in units])
        self.a, self.b = s_lo, max(s_hi - s_lo, 1e-6)

    def _stat(self, r, c):
        res = self._z(r) - self._cov_design(c) @ self.coef
        t2 = np.maximum(np.einsum('ij,jk,ik->i', res, self.Pinv, res), 0)
        # The chart statistic is quadratic in the residual, so it stays pinned
        # near zero for the first half of life and then explodes -- which makes
        # the exponential first-passage fit ill-posed on early truncations.
        # Variance-stabilising transforms are offered and picked on dev.
        if self.stat == 't2':
            return t2
        if self.stat == 'sqrt':
            return np.sqrt(t2)
        if self.stat == 'q4':
            return t2 ** 0.25
        if self.stat == 'log':
            return np.log1p(t2)
        raise ValueError(self.stat)

    def predict(self, r, c):
        return self._smooth((self._stat(r, c) - self.a) / self.b)


REGISTRY = {'ours': OursMLP, 'lr': LinReg, 'bspline': BSplineHI,
            'llke': LLKE, 'cabn': CaBN}


def build(name, **hp):
    return REGISTRY[name](**hp)


# --------------------------------------------------------------------------- #
#  data loading                                                                #
# --------------------------------------------------------------------------- #
def gp_stage_cost():
    """Measured cost of the SHARED upstream stage (Vecchia MOGP normal model),
    parsed from rul_r1_results.txt so the numbers keep their provenance.

    Returns per-seed means, in seconds:
      fit      GP training (16384 candidates -> 5 % detcov cleaning -> 8192 pts)
      dev_inf  point statistics of the 9 dev units  (~132,600 points)
      test_inf point statistics of the 6 test units (~ 87,600 points)

    Every HI method in the benchmark consumes the same residual table, so this
    cost is identical for all rows -- which is the point: it dominates."""
    fit, dev, test = [], [], []
    with open(os.path.join(HERE, 'rul_r1_results.txt')) as f:
        cur = {}
        for ln in f:
            if 'final model rebuilt' in ln:
                cur['fit'] = float(ln.split('(')[1].split('s)')[0])
            elif 'dev point stats done' in ln:
                cur['dev'] = float(ln.split('(')[1].split('s)')[0])
            elif 'test point stats done' in ln:
                cur['test'] = float(ln.split('(')[1].split('s)')[0])
                fit.append(cur['fit'])
                dev.append(cur['dev'] - cur['fit'])
                test.append(cur['test'] - cur['dev'])
    return float(np.mean(fit)), float(np.mean(dev)), float(np.mean(test))


def timeit(fn, warmup=2, reps=5):
    """warm-up then median-of-reps wall time, in seconds."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def load_split(sd, split, variant='trim'):
    """-> units, R (raw residual rows), cov (flight hours, (n,1))"""
    z = np.load(os.path.join(HERE, f'rul_input_s{sd}.npz'))
    units = sorted({int(k.split('_')[1]) for k in z.files
                    if k.startswith(f'{split}_') and k.endswith('_ucyc')})
    R = {u: np.asarray(z[f'{split}_{u}_{variant}'], float) for u in units}
    cov = {u: np.asarray(z[f'{split}_{u}_hours'], float)[:, None] for u in units}
    return units, R, cov
