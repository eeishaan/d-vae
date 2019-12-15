import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import pickle
import numpy as np
import os.path
from math import floor
import time
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel())
        self.covar_module = InducingPointKernel(
            self.base_covar_module,
            inducing_points=train_x[:500, :],
            likelihood=likelihood,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


def eval(model, likelihood, test_x, test_y, split="val"):
    model.eval()
    likelihood.eval()
    with gpytorch.settings.max_preconditioner_size(10), torch.no_grad():
        with gpytorch.settings.use_toeplitz(
            False
        ), gpytorch.settings.max_root_decomposition_size(
            30
        ), gpytorch.settings.fast_pred_var():
            preds = model(test_x)

    MAE = torch.mean(torch.abs(preds.mean - test_y))
    print("{} MAE: {}".format(split, MAE))
    # checkpoint = torch.load(PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    return MAE


def normalize_data(y, mean, std):
    y = (y - mean) / std
    return y


def load_data(device=None):
    y_train = np.concatenate(pickle.load(open(f"../train_accs.pkl", "rb")), axis=0)

    X_train = np.concatenate(
        pickle.load(open(f"../train_latent_rep.pkl", "rb")), axis=0
    )

    y_val = np.concatenate(pickle.load(open(f"../val_accs.pkl", "rb")), axis=0)
    X_val = np.concatenate(pickle.load(open(f"../val_latent_rep.pkl", "rb")), axis=0)

    y_test = np.concatenate(pickle.load(open(f"../test_accs.pkl", "rb")), axis=0)
    X_test = np.concatenate(pickle.load(open(f"../test_latent_rep.pkl", "rb")), axis=0)

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device).to(dtype=torch.float)
    X_val = torch.from_numpy(X_val).to(device)
    y_val = torch.from_numpy(y_val).to(device).to(dtype=torch.float)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device).to(dtype=torch.float)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def main():
    # set the seed
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_data, val_data, test_data = load_data(device)
    X_train, y_train = train_data
    X_val, y_val = val_data
    X_test, y_test = test_data

    n = len(X_train)  # The number of data points
    print(n)
    mean = torch.mean(y_train)
    std = torch.std(y_train)

    print(mean, std)

    y_train = normalize_data(y_train, mean, std)
    y_test = normalize_data(y_test, mean, std)
    y_val = normalize_data(y_val, mean, std)

    # train_dataset = TensorDataset(X_train, y_train)
    # val_dataset = TensorDataset(X_val, y_val)
    # test_dataset = TensorDataset(X_test, y_test)

    # train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)
    model = GPRegressionModel(X_train, y_train, likelihood).to(device=device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    epochs = 100
    best_mae = float("inf")
    with gpytorch.settings.max_cg_iterations(45):
        for i in range(epochs):
            model.train()
            likelihood.train()

            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(X_train)
            # Calc loss and backprop derivatives
            loss = -mll(output, y_train)
            loss.backward(retain_graph=True)
            optimizer.step()

            print("Iter %d/%d - Loss: %.3f" % (i + 1, epochs, loss.item()))
            mae = eval(model, likelihood, X_val, y_val, split="val")
            if mae < best_mae:
                best_mae = mae
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "likelihood": likelihood.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss,
                        "mae": mae,
                    },
                    "./best_sgp_model.pt",
                )

    eval(model, likelihood, X_test, y_test, split="test")


if __name__ == "__main__":
    main()
