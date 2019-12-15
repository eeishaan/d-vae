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
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy
from scipy import stats


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


def RMSELoss(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def eval(model, likelihood, test_loader, split="val"):
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.0])
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
    means = means[1:]
    return means


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = WhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    inducing_points = X_train[:500, :]
    model = GPModel(inducing_points=inducing_points).to(device=device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device=device)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
    mll = gpytorch.mlls.VariationalELBO(
        likelihood, model, num_data=y_train.size(0), combine_terms=False
    )

    epochs = 100

    for i in range(epochs):
        model.train()
        likelihood.train()
        total_loss = 0.0
        for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
            start_time = time.time()
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model(x_batch)
            # Calc loss and backprop derivatives
            log_lik, kl_div, log_prior = mll(output, y_batch)
            loss = -(log_lik - kl_div + log_prior)

            loss.backward(retain_graph=True)
            optimizer.step()
            total_loss += loss.item()
            print(
                "Epoch %d [%d/%d] - Loss: %.3f [%.3f, %.3f, %.3f]"
                % (
                    i + 1,
                    minibatch_i,
                    len(train_loader),
                    loss.item(),
                    log_lik.item(),
                    kl_div.item(),
                    log_prior.item(),
                )
            )

        # torch.cuda.empty_cache()
        print("Iter %d/%d - Loss: %.3f" % (i + 1, epochs, total_loss))
        means = eval(model, likelihood, val_loader, split="val")
        mae = torch.mean(torch.abs(means - y_val.cpu()))
        pearsonr = stats.pearsonr(means.cpu().numpy(), y_val.cpu().numpy())[0]
        rmse = RMSELoss(means, y_val)
        print("Val | MAE: {}| pearsonr: {} | rmse: {}".format(mae, pearsonr, rmse))

    means = eval(model, likelihood, test_loader, split="test")
    mae = torch.mean(torch.abs(means - y_test.cpu()))
    pearsonr = stats.pearsonr(means.cpu().numpy(), y_test.cpu().numpy())[0]
    rmse = RMSELoss(means, y_test)
    print("Val | MAE: {}| pearsonr: {} | rmse: {}".format(mae, pearsonr, rmse))


if __name__ == "__main__":
    main()
