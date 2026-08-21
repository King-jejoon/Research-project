"""Train a GPyTorch multi-output GP with grouped Vecchia likelihood.

The GPyTorch model supplies the mean, latent multi-task covariance, and
observation-noise parameters.  The likelihood objective is the MMD-ordered,
previous-neighbour, automatically grouped Vecchia approximation implemented in
``vecchia_mmd.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import gpytorch

from vecchia_mmd import (
    VecchiaStructure,
    build_vecchia_structure,
    multioutput_conditional_mean,
    multioutput_conditional_test_log_likelihood,
    multioutput_vecchia_log_likelihood,
)


Tensor = torch.Tensor


class GPyTorchMOGP(gpytorch.Module):
    """Intrinsic coregionalization multi-output GP.

    This is a lightweight GPyTorch module, not an ``ExactGP`` subclass, because
    we train it with a custom Vecchia objective rather than GPyTorch's exact
    marginal log likelihood.
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        rank: int = 1,
        kernel: str = "matern32",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tasks = num_tasks

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=num_tasks,
        )
        self.base_kernel = _make_base_kernel(kernel, input_dim=input_dim)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.ScaleKernel(self.base_kernel),
            num_tasks=num_tasks,
            rank=rank,
        )

    def forward(self, x: Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
        mean = self.mean_module(x)
        cov = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, cov)

    def mean(self, x: Tensor) -> Tensor:
        return self.mean_module(x)

    def latent_covariance(self, x1: Tensor, x2: Tensor) -> Tensor:
        return self.covar_module(x1, x2).to_dense()

    def observation_covariance(
        self,
        likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
        x1: Tensor,
        x2: Tensor,
    ) -> Tensor:
        """Dense covariance of observed outputs.

        Vecchia likelihood evaluations request square covariance blocks over one
        ordered location set ``U``.  In that case we add the GPyTorch likelihood
        noise.  For non-square calls this falls back to the latent covariance.
        """
        if x1.shape == x2.shape and torch.equal(x1, x2):
            latent = self.forward(x1)
            observed = likelihood(latent)
            return observed.lazy_covariance_matrix.to_dense()
        return self.latent_covariance(x1, x2)


@dataclass
class VecchiaTrainingResult:
    structure: VecchiaStructure
    losses: list[float]
    log_likelihoods: list[float]


def train_vecchia_mogp(
    model: GPyTorchMOGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_x: Tensor,
    train_y: Tensor,
    m: int,
    num_steps: int = 200,
    lr: float = 0.05,
    group: bool = True,
    center: Optional[Tensor] = None,
    jitter: float = 1e-6,
    optimizer_factory: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
    verbose: bool = True,
) -> VecchiaTrainingResult:
    """Optimize GPyTorch MOGP hyperparameters using grouped Vecchia likelihood."""
    if train_y.ndim != 2:
        raise ValueError("train_y must have shape (n, num_tasks)")
    if train_y.shape[1] != model.num_tasks:
        raise ValueError("train_y task dimension must match model.num_tasks")

    structure = build_vecchia_structure(train_x, m=m, group=group, center=center)
    optimizer = optimizer_factory(
        list(model.parameters()) + list(likelihood.parameters()),
        lr=lr,
    )

    losses: list[float] = []
    log_likelihoods: list[float] = []
    model.train()
    likelihood.train()

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        log_likelihood = vecchia_mogp_log_likelihood(
            model,
            likelihood,
            train_x,
            train_y,
            structure,
            jitter=jitter,
        )
        loss = -log_likelihood
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach()))
        log_likelihoods.append(float(log_likelihood.detach()))
        if verbose and (step == 1 or step == num_steps or step % max(1, num_steps // 10) == 0):
            print(f"step {step:4d}/{num_steps}: vecchia log likelihood = {float(log_likelihood.detach()):.6f}")

    return VecchiaTrainingResult(
        structure=structure,
        losses=losses,
        log_likelihoods=log_likelihoods,
    )


def vecchia_mogp_log_likelihood(
    model: GPyTorchMOGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_x: Tensor,
    train_y: Tensor,
    structure: VecchiaStructure,
    jitter: float = 1e-6,
) -> Tensor:
    """Grouped Vecchia approximate log likelihood for a GPyTorch MOGP."""
    return multioutput_vecchia_log_likelihood(
        y=train_y,
        x=train_x,
        structure=structure,
        covariance_fn=lambda a, b: model.observation_covariance(likelihood, a, b),
        mean=lambda a: model.mean(a),
        jitter=jitter,
    )


def vecchia_mogp_conditional_log_likelihood(
    model: GPyTorchMOGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    train_structure: VecchiaStructure,
    m: int,
    test_order: Optional[Tensor] = None,
    group_tests: bool = False,
    jitter: float = 1e-6,
) -> Tensor:
    """Approximate ``log p(test_y | train_y)`` under the fitted MOGP."""
    return multioutput_conditional_test_log_likelihood(
        y_train=train_y,
        x_train=train_x,
        y_test=test_y,
        x_test=test_x,
        train_structure=train_structure,
        covariance_fn=lambda a, b: model.observation_covariance(likelihood, a, b),
        m=m,
        test_order=test_order,
        group_tests=group_tests,
        mean_train=lambda a: model.mean(a),
        mean_test=lambda a: model.mean(a),
        jitter=jitter,
    )


def vecchia_mogp_predictive_mean(
    model: GPyTorchMOGP,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_x: Tensor,
    train_y: Tensor,
    test_x: Tensor,
    train_structure: VecchiaStructure,
    m: int,
    test_order: Optional[Tensor] = None,
    group_tests: bool = False,
    jitter: float = 1e-6,
    return_ordered: bool = False,
) -> Tensor:
    """Approximate conditional mean ``E[Y_test | Y_train]`` as predictions."""
    return multioutput_conditional_mean(
        y_train=train_y,
        x_train=train_x,
        x_test=test_x,
        train_structure=train_structure,
        covariance_fn=lambda a, b: model.observation_covariance(likelihood, a, b),
        m=m,
        test_order=test_order,
        group_tests=group_tests,
        mean_train=lambda a: model.mean(a),
        mean_test=lambda a: model.mean(a),
        jitter=jitter,
        return_ordered=return_ordered,
    )


def _make_base_kernel(kernel: str, input_dim: int) -> gpytorch.kernels.Kernel:
    kernel = kernel.lower()
    if kernel in {"rbf", "sqexp", "squared_exponential"}:
        return gpytorch.kernels.RBFKernel(ard_num_dims=input_dim)
    if kernel in {"matern12", "exponential"}:
        return gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=input_dim)
    if kernel in {"matern32", "matern3/2"}:
        return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=input_dim)
    if kernel in {"matern52", "matern5/2"}:
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=input_dim)
    raise ValueError(f"unknown kernel {kernel!r}")
