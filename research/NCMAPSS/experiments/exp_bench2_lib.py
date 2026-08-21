"""exp_bench2_lib.py — benchmark of NORMAL MODELS (the MOGP slot).

This is the benchmark on the slide: CaBN (Wei et al. 2025) / LR / LLKE /
B-spline compete with the Vecchia MOGP at modelling the healthy conditional
distribution of the 5 sensors given the 4 operating parameters.

    W = [alt, Mach, TRA, T2]  ->  X = [T30, T48, T50, Nc, Wf]

Every method must supply exactly what the frozen downstream consumes, with the
same definitions as demo_cond2.conditional_stats2:

    pred    conditional mean, returned in RAW sensor units (for the residual)
    detcov  -0.5 * logdet S   (S = predictive covariance, standardised space)
    ll      detcov - 0.5 * m2 - 0.5 * q * log(2 pi),   m2 = r^T S^-1 r

Everything after this block -- detcov cleaning of the training set, the
inspection gate V*, per-flight trim25 aggregation, the HI network and the
first-passage RUL -- stays byte-identical to the frozen pipeline.

Methods
  lr       X = beta + B^T w + eps,  eps ~ N(0, Sigma)          (full Sigma)
  bspline  additive cubic B-spline basis in the 4 inputs
  llke     local linear kernel estimation, product Gaussian kernel
  cabn     Wei et al. (2025), K = 1: mean beta + B^T w, residual DAG
           R = W^T R + eps -> Phi = (I-W^T)^-1 Sigma (I-W)^-1
  (mogp)   the frozen Vecchia MOGP; not re-implemented here, its residual
           tables are the existing rul_input_s*.npz
"""
import os, sys, time
import numpy as np
from scipy.interpolate import BSpline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp_bench_hi import _notears

Q = 5
LOG2PI = float(np.log(2 * np.pi))


# --------------------------------------------------------------------------- #
#  shared Gaussian statistics                                                  #
# --------------------------------------------------------------------------- #
def gauss_stats(res_z, S_logdet, S_inv, ):
    """res_z (n,q) standardised residual; S_logdet (n,), S_inv (n,q,q) or (q,q)"""
    detcov = -0.5 * S_logdet
    if S_inv.ndim == 2:
        m2 = np.einsum('ij,jk,ik->i', res_z, S_inv, res_z)
    else:
        m2 = np.einsum('ij,ijk,ik->i', res_z, S_inv, res_z)
    return detcov, detcov - 0.5 * m2 - 0.5 * Q * LOG2PI


class NormalModel:
    """fit(W, X) on raw arrays; stats(Wq, Xq) -> pred(raw), detcov, ll"""
    name = 'base'

    def __init__(self, **hp):
        self.hp = hp

    def _scale_fit(self, W, X):
        self.xs = StandardScaler().fit(W)
        self.ys = StandardScaler().fit(X)
        return self.xs.transform(W), self.ys.transform(X)

    def _unscale(self, Yz):
        return Yz * self.ys.scale_ + self.ys.mean_

    def fit(self, W, X):
        raise NotImplementedError

    def stats(self, Wq, Xq, chunk=8192):
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  linear-in-basis models (lr / bspline / cabn share the machinery)            #
# --------------------------------------------------------------------------- #
class _LinearBasis(NormalModel):
    """Y = D(w) @ C + E.  Predictive covariance S = Sigma * (1 + leverage)."""

    def _design(self, Wz):
        raise NotImplementedError

    def _resid_cov(self, E):
        return np.cov(E.T) + 1e-10 * np.eye(Q)

    def fit(self, W, X):
        Wz, Xz = self._scale_fit(W, X)
        self._prepare(Wz)
        D = self._design(Wz)
        self.C = np.linalg.lstsq(D, Xz, rcond=None)[0]
        E = Xz - D @ self.C
        self.Sigma = self._resid_cov(E)
        self.DtDinv = np.linalg.pinv(D.T @ D)
        self.S_inv = np.linalg.inv(self.Sigma)
        self.S_logdet = float(np.linalg.slogdet(self.Sigma)[1])

    def _prepare(self, Wz):
        pass

    def stats(self, Wq, Xq, chunk=8192):
        preds, dcs, lls = [], [], []
        for i in range(0, len(Wq), chunk):
            Wz = self.xs.transform(Wq[i:i + chunk])
            Xz = self.ys.transform(Xq[i:i + chunk])
            D = self._design(Wz)
            mu = D @ self.C
            h = np.einsum('ij,jk,ik->i', D, self.DtDinv, D)   # leverage
            f = 1.0 + np.maximum(h, 0.0)
            logdet = self.S_logdet + Q * np.log(f)
            res = Xz - mu
            detcov = -0.5 * logdet
            m2 = np.einsum('ij,jk,ik->i', res, self.S_inv, res) / f
            preds.append(self._unscale(mu))
            dcs.append(detcov)
            lls.append(detcov - 0.5 * m2 - 0.5 * Q * LOG2PI)
        return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


class LinearNM(_LinearBasis):
    name = 'lr'

    def _design(self, Wz):
        return np.column_stack([np.ones(len(Wz)), Wz])


class BSplineNM(_LinearBasis):
    name = 'bspline'

    def __init__(self, n_knots=5, **hp):
        super().__init__(n_knots=n_knots, **hp)
        self.n_knots = n_knots

    def _prepare(self, Wz):
        q = np.linspace(0, 1, self.n_knots + 2)[1:-1]
        self.knots = []
        for j in range(Wz.shape[1]):
            lo, hi = Wz[:, j].min(), Wz[:, j].max()
            pad = 0.05 * (hi - lo) + 1e-6
            self.knots.append(np.r_[[lo - pad] * 4, np.quantile(Wz[:, j], q),
                                    [hi + pad] * 4])

    def _design(self, Wz):
        cols = [np.ones(len(Wz))]
        for j in range(Wz.shape[1]):
            t = self.knots[j]
            x = np.clip(Wz[:, j], t[3] + 1e-9, t[-4] - 1e-9)
            cols.append(BSpline.design_matrix(x, t, 3).toarray()[:, 1:])
        return np.column_stack(cols)


class CaBNNM(_LinearBasis):
    """K=1 CaBN: covariate-adjusted mean + sparse DAG on the residual vector.
    Sigma is NOT the empirical covariance but Phi = (I-W^T)^-1 diag(s) (I-W)^-1,
    the covariance implied by the estimated Bayesian network."""
    name = 'cabn'

    def __init__(self, lam1=0.01, **hp):
        super().__init__(lam1=lam1, **hp)
        self.lam1 = lam1

    def _design(self, Wz):
        return np.column_stack([np.ones(len(Wz)), Wz])

    def _resid_cov(self, E):
        Wadj = _notears(E, lam1=self.lam1)
        eps = E - E @ Wadj
        s = eps.var(0) + 1e-8
        M = np.eye(Q) - Wadj
        Minv = np.linalg.inv(M)
        self.Wadj = Wadj
        return Minv.T @ np.diag(s) @ Minv + 1e-10 * np.eye(Q)


# --------------------------------------------------------------------------- #
#  local linear kernel estimation                                              #
# --------------------------------------------------------------------------- #
class LLKENM(NormalModel):
    """Local linear fit of the 5 outputs on the 4 inputs with a product
    Gaussian kernel.  Predictive covariance = Sigma * (1 + ||l(w)||^2), the
    usual local-linear smoother variance factor."""
    name = 'llke'

    def __init__(self, h=0.5, ridge=1e-6, **hp):
        super().__init__(h=h, ridge=ridge, **hp)
        self.h = h
        self.ridge = ridge

    def fit(self, W, X):
        self.Wz, self.Xz = self._scale_fit(W, X)
        # residual covariance from the smoother's own fit on a subsample
        idx = np.random.default_rng(0).choice(len(self.Wz),
                                              min(1024, len(self.Wz)), replace=False)
        mu, _ = self._local_fit(self.Wz[idx])
        E = self.Xz[idx] - mu
        self.Sigma = np.cov(E.T) + 1e-10 * np.eye(Q)
        self.S_inv = np.linalg.inv(self.Sigma)
        self.S_logdet = float(np.linalg.slogdet(self.Sigma)[1])

    def _local_fit(self, Wq, chunk=256):
        """-> mu (n,q), lnorm2 (n,) = ||equivalent kernel weights||^2"""
        mus, lns = [], []
        p = self.Wz.shape[1]
        I = self.ridge * np.eye(p + 1)
        for i in range(0, len(Wq), chunk):
            w = Wq[i:i + chunk]
            d = w[:, None, :] - self.Wz[None, :, :]            # (c, n, p)
            K = np.exp(-0.5 * (d / self.h) ** 2).prod(2)       # (c, n)
            D = np.concatenate([np.ones((*d.shape[:2], 1)), d], 2)   # (c, n, p+1)
            KD = K[:, :, None] * D
            A = np.einsum('cnj,cnk->cjk', KD, D) + I
            b = np.einsum('cnj,nq->cjq', KD, self.Xz)
            beta = np.linalg.solve(A, b)                        # (c, p+1, q)
            mus.append(beta[:, 0, :])
            Ainv_e1 = np.linalg.solve(A, np.tile(np.eye(p + 1)[:, :1],
                                                 (len(w), 1, 1)))[:, :, 0]
            l = np.einsum('cj,cnj->cn', Ainv_e1, KD)            # equivalent weights
            lns.append((l * l).sum(1))
        return np.concatenate(mus), np.concatenate(lns)

    def stats(self, Wq, Xq, chunk=8192):
        preds, dcs, lls = [], [], []
        for i in range(0, len(Wq), chunk):
            Wz = self.xs.transform(Wq[i:i + chunk])
            Xz = self.ys.transform(Xq[i:i + chunk])
            mu, ln2 = self._local_fit(Wz)
            f = 1.0 + ln2
            logdet = self.S_logdet + Q * np.log(f)
            res = Xz - mu
            detcov = -0.5 * logdet
            m2 = np.einsum('ij,jk,ik->i', res, self.S_inv, res) / f
            preds.append(self._unscale(mu))
            dcs.append(detcov)
            lls.append(detcov - 0.5 * m2 - 0.5 * Q * LOG2PI)
        return np.concatenate(preds), np.concatenate(dcs), np.concatenate(lls)


REGISTRY = {'lr': LinearNM, 'bspline': BSplineNM, 'llke': LLKENM, 'cabn': CaBNNM}


def build(name, **hp):
    return REGISTRY[name](**hp)
