"""demo_cond2.py — CORRECTED port of the DEMO conditional statistics.

Fixes a math bug in the original (demo_cond.py / vecchia_mmd.py lines 433-436):
the quadratic term of the conditional log-likelihood used

    A = L22 @ r ;  quad = -0.5 * (A.T @ A)        # = -0.5 * r^T (L22^T L22) r

i.e. the residual is weighted BY the conditional covariance instead of by its
inverse.  The correct Gaussian conditional log-density quadratic is

    quad = -0.5 * r^T S^{-1} r  with  S = L22 L22^T
         = -0.5 * || L22^{-1} r ||^2               # solve_triangular, not matmul

Consequences of the bug: residuals in high-uncertainty regions (extreme W) are
AMPLIFIED and residuals where the model is confident are SHRUNK -- exactly the
false-alarm / late-detection instability seen in end2end_v2.

Returns per test point:
  pred    (n,q)  Vecchia conditional predictive mean       (unchanged)
  detcov  (n,)   -sum(log diag L22) = -0.5*logdet(S)        (unchanged)
  m2      (n,)   Mahalanobis^2 = r^T S^{-1} r  (>=0, ~chi2(q) in-control)
  ll_fix  (n,)   detcov - 0.5*m2 - 0.5*q*log(2*pi)          (corrected LL)
  ll_orig (n,)   original (buggy) LL, kept for comparison
"""
import torch
import math
from vecchia_mmd import (
    mmd_ordering, _test_neighbours, _automatic_grouping_subset,
    _effective_group_neighbours, flatten_multioutput_covariance,
    _stabilize_covariance, _as_output_matrix,
)


def _mean_mat(model, x, q):
    v = model.mean(x)
    if v.ndim == 2 and v.shape == (x.shape[0], q):
        return v
    if v.ndim == 1 and v.shape[0] == q:
        return v.unsqueeze(0).expand(x.shape[0], q)
    if v.ndim == 0:
        return v.expand(x.shape[0], q)
    return v.reshape(x.shape[0], q)


def conditional_stats2(model, likelihood, train_x, train_y, structure, test_x, m,
                       test_y=None, jitter=1e-6):
    ty = _as_output_matrix(train_y)
    q = ty.shape[1]
    n_train = train_x.shape[0]; n_test = test_x.shape[0]

    x_train_ord = structure.ordered(train_x)
    y_train_ord = structure.ordered(ty)
    mu_train_ord = _mean_mat(model, train_x, q).index_select(0, structure.order)

    test_order = mmd_ordering(test_x)
    x_test_ord = test_x.index_select(0, test_order)
    mu_test_ord = _mean_mat(model, test_x, q).index_select(0, test_order)
    y_test_ord = None if test_y is None else _as_output_matrix(test_y).index_select(0, test_order)

    x_all = torch.cat([x_train_ord, x_test_ord], 0)
    mu_all = torch.cat([mu_train_ord, mu_test_ord], 0)
    values_all = torch.empty(n_train + n_test, q, dtype=ty.dtype, device=ty.device)
    values_all[:n_train] = y_train_ord
    values_all[n_train:] = mu_test_ord

    test_neighbours = _test_neighbours(x_all, n_train=n_train, n_test=n_test, m=m)
    neighbours = structure.neighbours + test_neighbours
    groups = _automatic_grouping_subset(neighbours, range(n_train, n_train + n_test), m=m)
    eff = _effective_group_neighbours(neighbours, groups, device=x_all.device)
    covfn = lambda a, b: model.observation_covariance(likelihood, a, b)

    detcov = torch.zeros(n_test, dtype=ty.dtype, device=ty.device)
    m2 = torch.zeros(n_test, dtype=ty.dtype, device=ty.device)
    quad_orig = torch.zeros(n_test, dtype=ty.dtype, device=ty.device)
    for i in range(n_train, n_train + n_test):
        ji = eff[i]; cond = ji[ji != i]
        li = i - n_train
        if cond.numel() == 0:
            values_all[i] = mu_all[i]
            continue
        query = torch.cat([cond, torch.tensor([i], dtype=torch.long, device=x_all.device)])
        k = flatten_multioutput_covariance(covfn(x_all.index_select(0, query), x_all.index_select(0, query)))
        cs = int(cond.numel()) * q
        k = _stabilize_covariance(k, jitter=jitter)
        centered = (values_all.index_select(0, cond) - mu_all.index_select(0, cond)).reshape(-1, 1)
        chol = torch.linalg.cholesky(k)
        L11 = chol[:cs, :cs]; L21 = chol[cs:, :cs]; L22 = chol[cs:, cs:]
        alpha = torch.linalg.solve_triangular(L11, centered, upper=False)
        values_all[i] = mu_all[i] + (L21 @ alpha).reshape(q)
        detcov[li] = -torch.log(torch.diagonal(L22)).sum()
        if y_test_ord is not None:
            r = (values_all[i] - y_test_ord[li]).reshape(-1, 1)
            # ---- FIX: whiten by L22^{-1} (precision), not multiply by L22 ----
            w = torch.linalg.solve_triangular(L22, r, upper=False)
            m2[li] = (w * w).sum()
            # original buggy quad, kept for A/B comparison
            Ao = L22 @ r
            quad_orig[li] = -0.5 * (Ao * Ao).sum()

    inv = torch.empty_like(test_order); inv[test_order] = torch.arange(n_test, device=test_order.device)
    pred = values_all[n_train:].index_select(0, inv).detach()
    detcov = detcov.index_select(0, inv).detach()
    m2 = m2.index_select(0, inv).detach()
    quad_orig = quad_orig.index_select(0, inv).detach()
    c = 0.5 * q * math.log(2 * math.pi)
    ll_fix = (detcov - 0.5 * m2 - c).detach()
    ll_orig = (detcov + quad_orig - c).detach()
    return (pred.cpu().numpy(), detcov.cpu().numpy(), m2.cpu().numpy(),
            ll_fix.cpu().numpy(), ll_orig.cpu().numpy())
