"""Vecchia GP approximation with MMD ordering and automatic grouping.

The implementation follows the ordering and grouping rules in
"Permutation and Grouping Methods for Sharpening Gaussian Process
Approximations".  All indices exposed by this module are zero-based.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Optional, Union

import torch


Tensor = torch.Tensor
CovarianceFn = Callable[[Tensor, Tensor], Tensor]
MultiOutputCovarianceFn = Callable[[Tensor, Tensor], Tensor]
Mean = Union[float, Tensor, Callable[[Tensor], Tensor]]


def pairwise_squared_distance(x1: Tensor, x2: Tensor) -> Tensor:
    """Return the matrix with entries ||x1_i - x2_j||^2."""
    x1 = x1.unsqueeze(-2)
    x2 = x2.unsqueeze(-3)
    return ((x1 - x2) ** 2).sum(dim=-1)


def rbf_covariance(
    x1: Tensor,
    x2: Tensor,
    variance: Union[Tensor, float] = 1.0,
    lengthscale: Union[Tensor, float] = 1.0,
    nugget: Union[Tensor, float] = 0.0,
) -> Tensor:
    """Squared-exponential covariance for examples and tests."""
    dist2 = pairwise_squared_distance(x1 / lengthscale, x2 / lengthscale)
    cov = torch.as_tensor(variance, dtype=x1.dtype, device=x1.device) * torch.exp(-0.5 * dist2)
    return _add_nugget_if_square(cov, nugget)


def exponential_covariance(
    x1: Tensor,
    x2: Tensor,
    variance: Union[Tensor, float] = 1.0,
    range_: Union[Tensor, float] = 1.0,
    nugget: Union[Tensor, float] = 0.0,
) -> Tensor:
    """Matern covariance with smoothness nu=1/2."""
    dist = torch.cdist(x1 / range_, x2 / range_)
    cov = torch.as_tensor(variance, dtype=x1.dtype, device=x1.device) * torch.exp(-dist)
    return _add_nugget_if_square(cov, nugget)


def matern32_covariance(
    x1: Tensor,
    x2: Tensor,
    variance: Union[Tensor, float] = 1.0,
    range_: Union[Tensor, float] = 1.0,
    nugget: Union[Tensor, float] = 0.0,
) -> Tensor:
    """Matern covariance with smoothness nu=3/2."""
    dist = torch.cdist(x1 / range_, x2 / range_)
    sqrt3_dist = math.sqrt(3.0) * dist
    cov = torch.as_tensor(variance, dtype=x1.dtype, device=x1.device) * (1.0 + sqrt3_dist) * torch.exp(-sqrt3_dist)
    return _add_nugget_if_square(cov, nugget)


def matern52_covariance(
    x1: Tensor,
    x2: Tensor,
    variance: Union[Tensor, float] = 1.0,
    range_: Union[Tensor, float] = 1.0,
    nugget: Union[Tensor, float] = 0.0,
) -> Tensor:
    """Matern covariance with smoothness nu=5/2."""
    dist = torch.cdist(x1 / range_, x2 / range_)
    sqrt5_dist = math.sqrt(5.0) * dist
    cov = (
        torch.as_tensor(variance, dtype=x1.dtype, device=x1.device)
        * (1.0 + sqrt5_dist + 5.0 * dist.square() / 3.0)
        * torch.exp(-sqrt5_dist)
    )
    return _add_nugget_if_square(cov, nugget)


def separable_multioutput_covariance(
    x1: Tensor,
    x2: Tensor,
    spatial_covariance_fn: CovarianceFn,
    output_covariance: Tensor,
    noise_covariance: Optional[Tensor] = None,
) -> Tensor:
    """Separable multi-output covariance ``K_x(x1,x2) kron B``.

    Returns a flattened block matrix with rows ordered as
    ``(location_0 output_0, ..., location_0 output_q, location_1 output_0, ...)``.
    ``noise_covariance`` is added only when ``x1`` and ``x2`` represent the same
    ordered location set.
    """
    spatial = spatial_covariance_fn(x1, x2)
    block = spatial[:, None, :, None] * output_covariance[None, :, None, :]
    if noise_covariance is not None and x1.shape[0] == x2.shape[0]:
        block = block.clone()
        idx = torch.arange(x1.shape[0], device=x1.device)
        block[idx, :, idx, :] = block[idx, :, idx, :] + noise_covariance
    return flatten_multioutput_covariance(block)


def flatten_multioutput_covariance(cov: Tensor) -> Tensor:
    """Convert ``(n1, q, n2, q)`` covariance blocks to ``(n1*q, n2*q)``."""
    if cov.ndim == 2:
        return cov
    if cov.ndim != 4:
        raise ValueError("multi-output covariance must have shape (n1*q, n2*q) or (n1, q, n2, q)")
    n1, q1, n2, q2 = cov.shape
    if q1 != q2:
        raise ValueError("output dimensions in covariance blocks must match")
    return cov.permute(0, 1, 2, 3).reshape(n1 * q1, n2 * q2)


def mmd_ordering(x: Tensor, center: Optional[Tensor] = None) -> Tensor:
    """Exact maximum-minimum-distance ordering.

    The first point is closest to ``center`` (the coordinate mean by default).
    Each later point maximizes its current minimum distance to the selected set.
    Ties are resolved by PyTorch's deterministic ``argmax``/``argmin`` rule.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape (n, d)")
    n = x.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)

    if center is None:
        center = x.mean(dim=0)
    center_dist2 = ((x - center) ** 2).sum(dim=1)
    first = torch.argmin(center_dist2)

    order = torch.empty(n, dtype=torch.long, device=x.device)
    selected = torch.zeros(n, dtype=torch.bool, device=x.device)
    order[0] = first
    selected[first] = True

    min_dist2 = pairwise_squared_distance(x, x[first : first + 1]).squeeze(1)
    min_dist2[first] = -1.0

    for pos in range(1, n):
        candidate = torch.argmax(min_dist2)
        order[pos] = candidate
        selected[candidate] = True
        new_dist2 = pairwise_squared_distance(x, x[candidate : candidate + 1]).squeeze(1)
        min_dist2 = torch.minimum(min_dist2, new_dist2)
        min_dist2[selected] = -1.0
    return order


def previous_neighbours(x_ordered: Tensor, m: int) -> list[Tensor]:
    """Find the previous ``m`` Euclidean nearest neighbours for each ordered row.

    Returns a list ``J`` where ``J[i]`` is sorted increasingly and includes ``i``
    itself, matching the paper's convention.  Early rows contain fewer than
    ``m`` previous neighbours.
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    n = x_ordered.shape[0]
    result: list[Tensor] = []
    for i in range(n):
        count = min(m, i)
        if count == 0:
            result.append(torch.tensor([i], dtype=torch.long, device=x_ordered.device))
            continue
        dist2 = pairwise_squared_distance(x_ordered[i : i + 1], x_ordered[:i]).squeeze(0)
        nn = torch.topk(dist2, k=count, largest=False, sorted=False).indices
        ji = torch.cat([nn, torch.tensor([i], dtype=torch.long, device=x_ordered.device)])
        result.append(torch.sort(ji).values)
    return result


def automatic_grouping(neighbours: list[Tensor], m: int) -> list[Tensor]:
    """Greedy automatic grouping from the paper.

    ``neighbours[i]`` must include ``i`` and be in ordered index space.  The
    returned groups are sorted tensors of ordered indices.
    """
    n = len(neighbours)
    groups: list[set[int]] = [{i} for i in range(n)]
    group_of = list(range(n))
    union_neighbours: list[set[int]] = [set(_to_int_list(j)) for j in neighbours]

    for ell in range(m):
        for i in range(n):
            js = _to_int_list(neighbours[i])
            if ell >= len(js) - 1:
                continue
            k = group_of[i]
            kp = group_of[js[ell]]
            if k == kp or not groups[k] or not groups[kp]:
                continue
            merged_u = union_neighbours[k] | union_neighbours[kp]
            if len(merged_u) ** 2 <= len(union_neighbours[k]) ** 2 + len(union_neighbours[kp]) ** 2 and len(merged_u)<2000:
                groups[k].update(groups[kp])
                union_neighbours[k] = merged_u
                for idx in groups[kp]:
                    group_of[idx] = k
                groups[kp].clear()
                union_neighbours[kp].clear()

    device = neighbours[0].device if neighbours else torch.device("cpu")
    return [
        torch.tensor(sorted(group), dtype=torch.long, device=device)
        for group in groups
        if group
    ]


@dataclass(frozen=True)
class VecchiaStructure:
    """Reusable ordering, neighbour, and grouping structure."""

    order: Tensor
    neighbours: list[Tensor]
    groups: list[Tensor]

    @property
    def n(self) -> int:
        return int(self.order.numel())

    def ordered(self, values: Tensor) -> Tensor:
        return values.index_select(0, self.order)

    def inverse_order(self) -> Tensor:
        inv = torch.empty_like(self.order)
        inv[self.order] = torch.arange(self.n, device=self.order.device)
        return inv


def build_vecchia_structure(x: Tensor, m: int, group: bool = True, center: Optional[Tensor] = None) -> VecchiaStructure:
    """Build MMD order, previous-neighbour sets, and optional automatic groups."""
    order = mmd_ordering(x, center=center)
    x_ordered = x.index_select(0, order)
    neighbours = previous_neighbours(x_ordered, m)
    groups = automatic_grouping(neighbours, m) if group else [
        torch.tensor([i], dtype=torch.long, device=x.device) for i in range(x.shape[0])
    ]
    return VecchiaStructure(order=order, neighbours=neighbours, groups=groups)


def vecchia_log_likelihood(
    y: Tensor,
    x: Tensor,
    structure: VecchiaStructure,
    covariance_fn: CovarianceFn,
    mean: Mean = 0.0,
    jitter: float = 1e-6,
) -> Tensor:
    """Approximate training log likelihood for observations ``(x, y)``.

    ``covariance_fn`` is a PyTorch callable ``covariance_fn(x1, x2) -> K``.
    Kernel parameters captured by the callable remain differentiable.
    """
    x_ord = structure.ordered(x)
    y_ord = structure.ordered(_as_column(y))
    mu_ord = _mean_as_column(mean, x).index_select(0, structure.order)
    residual = y_ord - mu_ord
    return _grouped_log_likelihood(residual, x_ord, structure.neighbours, structure.groups, covariance_fn, jitter)


def conditional_test_log_likelihood(
    y_train: Tensor,
    x_train: Tensor,
    y_test: Tensor,
    x_test: Tensor,
    train_structure: VecchiaStructure,
    covariance_fn: CovarianceFn,
    m: int,
    test_order: Optional[Tensor] = None,
    group_tests: bool = False,
    mean_train: Mean = 0.0,
    mean_test: Mean = 0.0,
    jitter: float = 1e-6,
) -> Tensor:
    """Approximate ``log p(y_test | y_train)``.

    Training observations keep the supplied MMD order and are placed before the
    test observations, so every test conditional can use training points plus
    previous test points as candidates.  The returned likelihood contribution
    contains only rows for test observations.
    """
    x_train_ord = train_structure.ordered(x_train)
    y_train_ord = train_structure.ordered(_as_column(y_train))
    mu_train_ord = _mean_as_column(mean_train, x_train).index_select(0, train_structure.order)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    if test_order is None:
        test_order = mmd_ordering(x_test)
    x_test_ord = x_test.index_select(0, test_order)
    y_test_ord = _as_column(y_test).index_select(0, test_order)
    mu_test_ord = _mean_as_column(mean_test, x_test).index_select(0, test_order)

    x_all = torch.cat([x_train_ord, x_test_ord], dim=0)
    residual_all = torch.cat([y_train_ord - mu_train_ord, y_test_ord - mu_test_ord], dim=0)

    test_neighbours = _test_neighbours(x_all, n_train=n_train, n_test=n_test, m=m)
    neighbours = train_structure.neighbours + test_neighbours
    if group_tests:
        groups = _automatic_grouping_subset(neighbours, range(n_train, n_train + n_test), m=m)
    else:
        groups = [
            torch.tensor([n_train + i], dtype=torch.long, device=x_all.device)
            for i in range(n_test)
        ]

    return _grouped_log_likelihood(residual_all, x_all, neighbours, groups, covariance_fn, jitter)


def multioutput_vecchia_log_likelihood(
    y: Tensor,
    x: Tensor,
    structure: VecchiaStructure,
    covariance_fn: MultiOutputCovarianceFn,
    mean: Mean = 0.0,
    jitter: float = 1e-6,
) -> Tensor:
    """Approximate training log likelihood for ``q`` outputs per location.

    ``y`` must have shape ``(n, q)``.  For each ordered location ``i`` this
    computes the block conditional density
    ``p(Y_i | Y_j, j in previous neighbours of i)``.  With grouping, a group is
    evaluated from one covariance matrix over ``U_k``; the contribution includes
    all ``q`` scalar Cholesky rows for each location in ``B_k``, which is exactly
    the multivariate conditional density of the output vector at that location.
    """
    y_mat = _as_output_matrix(y)
    x_ord = structure.ordered(x)
    y_ord = structure.ordered(y_mat)
    mu_ord = _mean_as_output_matrix(mean, x, y_mat.shape[1]).index_select(0, structure.order)
    residual = y_ord - mu_ord
    return _grouped_multioutput_log_likelihood(
        residual,
        x_ord,
        structure.neighbours,
        structure.groups,
        covariance_fn,
        jitter,
    )


def multioutput_conditional_test_log_likelihood(
    y_train: Tensor,
    x_train: Tensor,
    y_test: Tensor,
    x_test: Tensor,
    train_structure: VecchiaStructure,
    covariance_fn: MultiOutputCovarianceFn,
    m: int,
    test_order: Optional[Tensor] = None,
    group_tests: bool = False,
    mean_train: Mean = 0.0,
    mean_test: Mean = 0.0,
    jitter: float = 1e-6,
) -> Tensor:
    """Approximate multi-output ``log p(Y_test | Y_train)``.

    The approximation appends ordered test locations after ordered training
    locations.  Test conditionals may use all training locations and previous
    test locations as neighbours.  Only test-location block rows contribute to
    the returned log likelihood.
    """
    y_train_mat = _as_output_matrix(y_train)
    y_test_mat = _as_output_matrix(y_test)
    if y_train_mat.shape[1] != y_test_mat.shape[1]:
        raise ValueError("train and test outputs must have the same q")
    q = y_train_mat.shape[1]

    x_train_ord = train_structure.ordered(x_train)
    y_train_ord = train_structure.ordered(y_train_mat)
    mu_train_ord = _mean_as_output_matrix(mean_train, x_train, q).index_select(0, train_structure.order)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    if test_order is None:
        test_order = mmd_ordering(x_test)
    x_test_ord = x_test.index_select(0, test_order)
    y_test_ord = y_test_mat.index_select(0, test_order)
    mu_test_ord = _mean_as_output_matrix(mean_test, x_test, q).index_select(0, test_order)

    x_all = torch.cat([x_train_ord, x_test_ord], dim=0)
    mu_all = torch.cat([mu_train_ord, mu_test_ord], dim=0)
    values_all = torch.empty(n_train + n_test, q, dtype=y_train_mat.dtype, device=y_train_mat.device)
    values_all[:n_train] = y_train_ord
    values_all[n_train:] = mu_test_ord

    residual_all = torch.cat([y_train_ord - mu_train_ord, y_test_ord - mu_test_ord], dim=0)

    test_neighbours = _test_neighbours(x_all, n_train=n_train, n_test=n_test, m=m)
    neighbours = train_structure.neighbours + test_neighbours

    groups = _automatic_grouping_subset(neighbours, range(n_train, n_train + n_test), m=m)

    residual_ordered = residual_all
    x_ordered = x_all

    effective_neighbours = _effective_group_neighbours(neighbours, groups, device=x_all.device)

    quadratic = []
    detcov = []
    for i in range(n_train, n_train + n_test):
        ji = effective_neighbours[i]
        cond = ji[ji != i]
        if cond.numel() == 0:
            values_all[i] = mu_all[i]
            continue
        query = torch.cat([cond, torch.tensor([i], dtype=torch.long, device=x_all.device)])
        k = flatten_multioutput_covariance(covariance_fn(x_all.index_select(0, query), x_all.index_select(0, query)))
        cond_size = int(cond.numel()) * q
        expected = cond_size + q
        if k.shape != (expected, expected):
            raise ValueError(
                f"multi-output covariance returned shape {tuple(k.shape)}, expected {(expected, expected)}")
        k = _stabilize_covariance(k, jitter=1e-6)
        centered = (values_all.index_select(0, cond) - mu_all.index_select(0, cond)).reshape(-1, 1)

        chol = torch.linalg.cholesky(k)
        L11 = chol[:cond_size, :cond_size]
        L21 = chol[cond_size:, :cond_size]
        alpha = torch.linalg.solve_triangular(L11, centered, upper=False)
        values_all[i] = mu_all[i] + (L21 @ alpha).reshape(q)

        L22 = chol[cond_size:, cond_size:]
        mu = values_all[i] - y_test_ord[i - n_train]
        A = L22 @ mu
        quadratic.append(-0.5 * (A.T @ A).reshape(-1))
        detcov.append(-1.0 * torch.log(torch.diagonal(L22)).sum())

    quadratic_ordered = torch.stack(quadratic).reshape(-1)
    detcov_ordered = torch.stack(detcov).reshape(-1)
    pred_ordered = values_all[n_train:]

    inv_test_order = torch.empty_like(test_order)
    inv_test_order[test_order] = torch.arange(n_test, device=test_order.device)

    pred = pred_ordered.index_select(0, inv_test_order).cpu().detach().numpy()
    quadratic = quadratic_ordered.index_select(0, inv_test_order).cpu().detach().numpy()
    detcov = detcov_ordered.index_select(0, inv_test_order).cpu().detach().numpy()

    del k, L22, L11, L21

    return pred, quadratic, detcov


def multioutput_conditional_mean(
    y_train: Tensor,
    x_train: Tensor,
    x_test: Tensor,
    train_structure: VecchiaStructure,
    covariance_fn: MultiOutputCovarianceFn,
    m: int,
    test_order: Optional[Tensor] = None,
    group_tests: bool = False,
    mean_train: Mean = 0.0,
    mean_test: Mean = 0.0,
    jitter: float = 1e-6,
    return_ordered: bool = False,
) -> Tensor:
    """Approximate conditional mean ``E[Y_test | Y_train]`` for q outputs.

    Test locations are appended after the ordered training locations.  The mean
    is computed recursively in test order: if a test location conditions on
    previous test locations, their already-computed conditional means are used.
    When ``group_tests=True``, the grouped effective neighbour set
    ``bar J_i = {j in U_k: j <= i}`` is used for each test location.
    """
    y_train_mat = _as_output_matrix(y_train)
    q = y_train_mat.shape[1]

    x_train_ord = train_structure.ordered(x_train)
    y_train_ord = train_structure.ordered(y_train_mat)
    mu_train_ord = _mean_as_output_matrix(mean_train, x_train, q).index_select(0, train_structure.order)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]
    if test_order is None:
        test_order = mmd_ordering(x_test)
    x_test_ord = x_test.index_select(0, test_order)
    mu_test_ord = _mean_as_output_matrix(mean_test, x_test, q).index_select(0, test_order)

    x_all = torch.cat([x_train_ord, x_test_ord], dim=0)
    mu_all = torch.cat([mu_train_ord, mu_test_ord], dim=0)
    values_all = torch.empty(n_train + n_test, q, dtype=y_train_mat.dtype, device=y_train_mat.device)
    values_all[:n_train] = y_train_ord
    values_all[n_train:] = mu_test_ord

    test_neighbours = _test_neighbours(x_all, n_train=n_train, n_test=n_test, m=m)
    neighbours = train_structure.neighbours + test_neighbours
    if group_tests:
        groups = _automatic_grouping_subset(neighbours, range(n_train, n_train + n_test), m=m)
    else:
        groups = [
            torch.tensor([n_train + i], dtype=torch.long, device=x_all.device)
            for i in range(n_test)
        ]
    effective_neighbours = _effective_group_neighbours(neighbours, groups, device=x_all.device)

    for i in range(n_train, n_train + n_test):
        ji = effective_neighbours[i]
        cond = ji[ji != i]
        if cond.numel() == 0:
            values_all[i] = mu_all[i]
            continue
        query = torch.cat([cond, torch.tensor([i], dtype=torch.long, device=x_all.device)])
        k = flatten_multioutput_covariance(covariance_fn(x_all.index_select(0, query), x_all.index_select(0, query)))
        cond_size = int(cond.numel()) * q
        expected = cond_size + q
        if k.shape != (expected, expected):
            raise ValueError(f"multi-output covariance returned shape {tuple(k.shape)}, expected {(expected, expected)}")
        k = _stabilize_covariance(k, jitter)
        k_cc = k[:cond_size, :cond_size]
        k_tc = k[cond_size:, :cond_size]
        centered = (values_all.index_select(0, cond) - mu_all.index_select(0, cond)).reshape(-1, 1)
        chol = torch.linalg.cholesky(k_cc)
        alpha = torch.cholesky_solve(centered, chol)
        values_all[i] = mu_all[i] + (k_tc @ alpha).reshape(q)

    pred_ordered = values_all[n_train:]
    if return_ordered:
        return pred_ordered
    inv_test_order = torch.empty_like(test_order)
    inv_test_order[test_order] = torch.arange(n_test, device=test_order.device)
    return pred_ordered.index_select(0, inv_test_order)


class VecchiaMMDGP:
    """Small convenience wrapper around the functional API."""

    def __init__(self, m: int, group: bool = True, center: Optional[Tensor] = None, jitter: float = 1e-6):
        self.m = m
        self.group = group
        self.center = center
        self.jitter = jitter
        self.structure: Optional[VecchiaStructure] = None

    def fit_structure(self, x: Tensor) -> VecchiaStructure:
        self.structure = build_vecchia_structure(x, self.m, group=self.group, center=self.center)
        return self.structure

    def log_likelihood(self, y: Tensor, x: Tensor, covariance_fn: CovarianceFn, mean: Mean = 0.0) -> Tensor:
        if self.structure is None:
            self.fit_structure(x)
        assert self.structure is not None
        return vecchia_log_likelihood(y, x, self.structure, covariance_fn, mean=mean, jitter=self.jitter)

    def conditional_log_likelihood(
        self,
        y_train: Tensor,
        x_train: Tensor,
        y_test: Tensor,
        x_test: Tensor,
        covariance_fn: CovarianceFn,
        mean_train: Mean = 0.0,
        mean_test: Mean = 0.0,
        test_order: Optional[Tensor] = None,
    ) -> Tensor:
        if self.structure is None:
            self.fit_structure(x_train)
        assert self.structure is not None
        return conditional_test_log_likelihood(
            y_train=y_train,
            x_train=x_train,
            y_test=y_test,
            x_test=x_test,
            train_structure=self.structure,
            covariance_fn=covariance_fn,
            m=self.m,
            test_order=test_order,
            group_tests=self.group,
            mean_train=mean_train,
            mean_test=mean_test,
            jitter=self.jitter,
        )


class VecchiaMMDMultiOutputGP(VecchiaMMDGP):
    """Convenience wrapper for multi-output observations with shape ``(n, q)``."""

    def log_likelihood(self, y: Tensor, x: Tensor, covariance_fn: MultiOutputCovarianceFn, mean: Mean = 0.0) -> Tensor:
        if self.structure is None:
            self.fit_structure(x)
        assert self.structure is not None
        return multioutput_vecchia_log_likelihood(y, x, self.structure, covariance_fn, mean=mean, jitter=self.jitter)

    def conditional_log_likelihood(
        self,
        y_train: Tensor,
        x_train: Tensor,
        y_test: Tensor,
        x_test: Tensor,
        covariance_fn: MultiOutputCovarianceFn,
        mean_train: Mean = 0.0,
        mean_test: Mean = 0.0,
        test_order: Optional[Tensor] = None,
    ) -> Tensor:
        if self.structure is None:
            self.fit_structure(x_train)
        assert self.structure is not None
        return multioutput_conditional_test_log_likelihood(
            y_train=y_train,
            x_train=x_train,
            y_test=y_test,
            x_test=x_test,
            train_structure=self.structure,
            covariance_fn=covariance_fn,
            m=self.m,
            test_order=test_order,
            group_tests=self.group,
            mean_train=mean_train,
            mean_test=mean_test,
            jitter=self.jitter,
        )

    def conditional_mean(
        self,
        y_train: Tensor,
        x_train: Tensor,
        x_test: Tensor,
        covariance_fn: MultiOutputCovarianceFn,
        mean_train: Mean = 0.0,
        mean_test: Mean = 0.0,
        test_order: Optional[Tensor] = None,
        return_ordered: bool = False,
    ) -> Tensor:
        if self.structure is None:
            self.fit_structure(x_train)
        assert self.structure is not None
        return multioutput_conditional_mean(
            y_train=y_train,
            x_train=x_train,
            x_test=x_test,
            train_structure=self.structure,
            covariance_fn=covariance_fn,
            m=self.m,
            test_order=test_order,
            group_tests=self.group,
            mean_train=mean_train,
            mean_test=mean_test,
            jitter=self.jitter,
            return_ordered=return_ordered,
        )


def _grouped_log_likelihood(
    residual_ordered: Tensor,
    x_ordered: Tensor,
    neighbours: list[Tensor],
    groups: Iterable[Tensor],
    covariance_fn: CovarianceFn,
    jitter: float,
) -> Tensor:
    total = residual_ordered.new_tensor(0.0)
    log2pi = residual_ordered.new_tensor(math.log(2.0 * math.pi))
    n = residual_ordered.shape[0]

    for group in groups:
        if group.numel() == 0:
            continue
        u = _union_sorted([neighbours[int(i)] for i in _to_int_list(group)], device=x_ordered.device)
        k = covariance_fn(x_ordered.index_select(0, u), x_ordered.index_select(0, u))
        k = _stabilize_covariance(k, jitter)
        chol = torch.linalg.cholesky(k)
        z = torch.linalg.solve_triangular(chol, residual_ordered.index_select(0, u), upper=False)
        positions = torch.searchsorted(u, group)
        if not torch.equal(u.index_select(0, positions), group):
            raise RuntimeError("Each group member must belong to its union-neighbour set")
        diag = torch.diagonal(chol).index_select(0, positions).unsqueeze(-1)
        z_g = z.index_select(0, positions)
        total = total - 0.5 * (z_g.square() + 2.0 * torch.log(diag) + log2pi).sum()
        if int(u[-1]) >= n:
            raise RuntimeError("Neighbour index out of bounds")
    return total


def _grouped_multioutput_log_likelihood(
    residual_ordered: Tensor,
    x_ordered: Tensor,
    neighbours: list[Tensor],
    groups: Iterable[Tensor],
    covariance_fn: MultiOutputCovarianceFn,
    jitter: float,
) -> Tensor:
    total = residual_ordered.new_tensor(0.0)
    log2pi = residual_ordered.new_tensor(math.log(2.0 * math.pi))
    n, q = residual_ordered.shape

    for group in groups:
        if group.numel() == 0:
            continue
        u = _union_sorted([neighbours[int(i)] for i in _to_int_list(group)], device=x_ordered.device)
        xu = x_ordered.index_select(0, u)
        k = flatten_multioutput_covariance(covariance_fn(xu, xu))
        expected = int(u.numel()) * q
        if k.shape != (expected, expected):
            raise ValueError(
                f"multi-output covariance returned shape {tuple(k.shape)}, expected {(expected, expected)}"
            )
        k = _stabilize_covariance(k, jitter)
        chol = torch.linalg.cholesky(k)
        residual_flat = residual_ordered.index_select(0, u).reshape(-1, 1)
        z = torch.linalg.solve_triangular(chol, residual_flat, upper=False)

        positions = torch.searchsorted(u, group)
        if not torch.equal(u.index_select(0, positions), group):
            raise RuntimeError("Each group member must belong to its union-neighbour set")
        flat_positions = (positions[:, None] * q + torch.arange(q, device=x_ordered.device)[None, :]).reshape(-1)
        diag = torch.diagonal(chol).index_select(0, flat_positions).unsqueeze(-1)
        z_g = z.index_select(0, flat_positions)
        total = total - 0.5 * (z_g.square() + 2.0 * torch.log(diag) + log2pi).sum()
        if int(u[-1]) >= n:
            raise RuntimeError("Neighbour index out of bounds")
    return total


def _test_neighbours(x_all: Tensor, n_train: int, n_test: int, m: int) -> list[Tensor]:
    neighbours: list[Tensor] = []
    device = x_all.device
    for local_i in range(n_test):
        i = n_train + local_i
        count = min(m, i)
        if count == 0:
            neighbours.append(torch.tensor([i], dtype=torch.long, device=device))
            continue
        dist2 = pairwise_squared_distance(x_all[i : i + 1], x_all[:i]).squeeze(0)
        nn = torch.topk(dist2, k=count, largest=False, sorted=False).indices
        ji = torch.cat([nn, torch.tensor([i], dtype=torch.long, device=device)])
        neighbours.append(torch.sort(ji).values)
    return neighbours


def _automatic_grouping_subset(neighbours: list[Tensor], indices: Iterable[int], m: int) -> list[Tensor]:
    """Automatic grouping for a subset, allowing neighbours outside the subset."""
    index_list = list(indices)
    if not index_list:
        return []
    device = neighbours[index_list[0]].device
    groups: list[set[int]] = [{i} for i in index_list]
    group_of = {i: pos for pos, i in enumerate(index_list)}
    union_neighbours: list[set[int]] = [set(_to_int_list(neighbours[i])) for i in index_list]

    for ell in range(m):
        for i in index_list:
            js = _to_int_list(neighbours[i])
            if ell >= len(js) - 1:
                continue
            j = js[ell]
            if j not in group_of:
                continue
            k = group_of[i]
            kp = group_of[j]
            if k == kp or not groups[k] or not groups[kp]:
                continue
            merged_u = union_neighbours[k] | union_neighbours[kp]
            if len(merged_u) ** 2 <= len(union_neighbours[k]) ** 2 + len(union_neighbours[kp]) ** 2:
                groups[k].update(groups[kp])
                union_neighbours[k] = merged_u
                for idx in groups[kp]:
                    group_of[idx] = k
                groups[kp].clear()
                union_neighbours[kp].clear()

    return [
        torch.tensor(sorted(group), dtype=torch.long, device=device)
        for group in groups
        if group
    ]


def _effective_group_neighbours(neighbours: list[Tensor], groups: Iterable[Tensor], device: torch.device) -> dict[int, Tensor]:
    effective: dict[int, Tensor] = {}
    for group in groups:
        if group.numel() == 0:
            continue
        u = _union_sorted([neighbours[int(i)] for i in _to_int_list(group)], device=device)
        for i in _to_int_list(group):
            effective[i] = u[u <= i]
    return effective


def _union_sorted(tensors: Iterable[Tensor], device: torch.device) -> Tensor:
    values: set[int] = set()
    for tensor in tensors:
        values.update(_to_int_list(tensor))
    return torch.tensor(sorted(values), dtype=torch.long, device=device)


def _as_column(y: Tensor) -> Tensor:
    if y.ndim == 1:
        return y.unsqueeze(-1)
    if y.ndim == 2 and y.shape[1] == 1:
        return y
    raise ValueError("y must have shape (n,) or (n, 1)")


def _as_output_matrix(y: Tensor) -> Tensor:
    if y.ndim == 1:
        return y.unsqueeze(-1)
    if y.ndim == 2:
        return y
    raise ValueError("multi-output y must have shape (n, q)")


def _mean_as_column(mean: Mean, x: Tensor) -> Tensor:
    if callable(mean):
        value = mean(x)
    else:
        value = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    if value.ndim == 0:
        value = value.expand(x.shape[0])
    return _as_column(value)


def _mean_as_output_matrix(mean: Mean, x: Tensor, q: int) -> Tensor:
    if callable(mean):
        value = mean(x)
    else:
        value = torch.as_tensor(mean, dtype=x.dtype, device=x.device)
    if value.ndim == 0:
        return value.expand(x.shape[0], q)
    if value.ndim == 1:
        if value.shape[0] == q:
            return value.unsqueeze(0).expand(x.shape[0], q)
        if q == 1 and value.shape[0] == x.shape[0]:
            return value.unsqueeze(-1)
    if value.ndim == 2 and value.shape == (x.shape[0], q):
        return value
    raise ValueError("mean must be scalar, shape (q,), shape (n, q), or a callable returning one of those")


def _add_nugget_if_square(cov: Tensor, nugget: Union[Tensor, float]) -> Tensor:
    if cov.shape[-2] == cov.shape[-1]:
        eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
        cov = cov + torch.as_tensor(nugget, dtype=cov.dtype, device=cov.device) * eye
    return cov


def _stabilize_covariance(cov: Tensor, jitter: float) -> Tensor:
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    return cov + torch.as_tensor(jitter, dtype=cov.dtype, device=cov.device) * eye


def _to_int_list(indices: Tensor) -> list[int]:
    return [int(v) for v in indices.detach().cpu().tolist()]
