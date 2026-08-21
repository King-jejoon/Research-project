# gp_residuals.py
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

output_names = ['T24', 'T30', 'T48', 'T50', 'P15', 'P2', 'P21', 'P24',
                'Ps30', 'P40', 'P50', 'Nf', 'Nc', 'Wf']

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(),
            num_tasks=num_tasks,
            rank=2,
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


def run_gp_residuals(
    device,
    x_dev_normal_scaled, y_dev_normal_scaled,
    x_dev_abnormal_scaled, x_test_abnormal_scaled,
    y_dev_mean, y_dev_std,
    W_dev, X_s_dev, dev_abnormal_idx,
    W_test, X_s_test, test_abnormal_idx,
    num_iters=100, lr=0.1
):
    dev_residuals  = []
    test_residuals = []

    num_tasks = y_dev_normal_scaled.shape[1]

    # 학습 데이터
    train_x = x_dev_normal_scaled.to(device)
    train_y = y_dev_normal_scaled.to(device)

    # likelihood, model, optimizer
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)
    model      = MultitaskGPModel(train_x, train_y, likelihood, num_tasks).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    mll        = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # 학습 루프
    model.train(); likelihood.train()
    for i in range(num_iters):
        optimizer.zero_grad()
        output = model(train_x)
        loss   = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print(f"Iter {i + 1}/{num_iters} - Loss: {loss.item():.3f}")

    # DEV abnormal 잔차
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds_dev = likelihood(model(x_dev_abnormal_scaled))
        mean_dev  = preds_dev.mean

    for j, name in enumerate(output_names):
        y_true_dev = torch.tensor(
            X_s_dev[dev_abnormal_idx['indices'], j],
            dtype=torch.float32, device=device
        )
        y_pred_dev   = mean_dev[:, j] * y_dev_std[j] + y_dev_mean[j]
        residual_dev = (y_true_dev - y_pred_dev).detach().cpu().numpy()

        dev_residuals.append({
            'name':     name,
            'dataset':  'DEV',
            'idx':      dev_abnormal_idx.copy(),
            'residual': residual_dev.copy(),
        })

        plt.figure(figsize=(8, 4))
        plt.plot(dev_abnormal_idx['indices'], residual_dev, marker='o', linestyle='-', alpha=0.7)
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f"DEV Residuals for {name}")
        plt.xlabel("Index")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # TEST abnormal 잔차
    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        preds_test = likelihood(model(x_test_abnormal_scaled))
        mean_test  = preds_test.mean

    for j, name in enumerate(output_names):
        y_true_test = torch.tensor(
            X_s_test[test_abnormal_idx['indices'], j],
            dtype=torch.float32, device=device
        )
        y_pred_test   = mean_test[:, j] * y_dev_std[j] + y_dev_mean[j]
        residual_test = (y_true_test - y_pred_test).detach().cpu().numpy()

        test_residuals.append({
            'name':     name,
            'dataset':  'TEST',
            'idx':      test_abnormal_idx.copy(),
            'residual': residual_test.copy(),
        })

        plt.figure(figsize=(8, 4))
        plt.plot(test_abnormal_idx['indices'], residual_test, marker='o', linestyle='-', alpha=0.7)
        plt.axhline(0, color='black', linewidth=1)
        plt.title(f"TEST Residuals for {name}")
        plt.xlabel("Index")
        plt.ylabel("Residual")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        'dev_residuals':  dev_residuals,
        'test_residuals': test_residuals,
        'model':          model,
        'likelihood':     likelihood,
    }