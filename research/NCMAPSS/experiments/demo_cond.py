"""Port of the DEMO notebook (cell 11) per-point conditional statistics for a
fitted GPyTorchMOGP + Vecchia structure.  Returns, for each test point:
  pred   (n,q)  Vecchia conditional predictive mean
  detcov (n,)   -sum(log diag(L22))  = -0.5*logdet(cond.cov); LARGE = low uncertainty
  ll     (n,)   per-point conditional log-likelihood (needs test_y)
Enables paper methods #2 (condition-likelihood detection) and #3 (uncertainty filter).
"""
import torch
from vecchia_mmd import (
    mmd_ordering, _test_neighbours, _automatic_grouping_subset,
    _effective_group_neighbours, flatten_multioutput_covariance,
    _stabilize_covariance, _as_output_matrix,
)
import math


def _mean_mat(model, x, q):
    """DEMO-notebook override: use model.mean(x) -> (n, q)."""
    v = model.mean(x)
    if v.ndim == 2 and v.shape == (x.shape[0], q):
        return v
    if v.ndim == 1 and v.shape[0] == q:
        return v.unsqueeze(0).expand(x.shape[0], q)
    if v.ndim == 0:
        return v.expand(x.shape[0], q)
    return v.reshape(x.shape[0], q)


def conditional_stats(model, likelihood, train_x, train_y, structure, test_x, m,
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
    quad = torch.zeros(n_test, dtype=ty.dtype, device=ty.device)
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
            r = values_all[i] - y_test_ord[li]        # pred - true (whitened by L22)
            A = L22 @ r
            quad[li] = -0.5 * (A * A).sum()

    inv = torch.empty_like(test_order); inv[test_order] = torch.arange(n_test, device=test_order.device)
    pred = values_all[n_train:].index_select(0, inv).detach()
    detcov = detcov.index_select(0, inv).detach()
    quad = quad.index_select(0, inv).detach()
    ll = (detcov + quad - 0.5 * q * math.log(2 * math.pi)).detach()
    return pred.cpu().numpy(), detcov.cpu().numpy(), ll.cpu().numpy()
