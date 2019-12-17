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
from sklearn.decomposition import PCA


def visul_2d(model, likelihood, X_train, data_type='BN', save_dir=None):
    print(
        "Generating grid points from the 2-dim latent space to visualize smoothness w.r.t. score"
    )
    # random_inputs = torch.randn(y_test.shape[0], nz).cuda()
    max_xy = 0.3
    x_range = np.arange(-max_xy, max_xy, 0.005)
    y_range = np.arange(max_xy, -max_xy, -0.005)
    n = len(x_range)
    x_range, y_range = np.meshgrid(x_range, y_range)
    x_range, y_range = x_range.reshape((-1, 1)), y_range.reshape((-1, 1))

    if True:  # select two principal components to visualize
        pca = PCA(n_components=2, whiten=True)
        pca.fit(X_train)
        d1, d2 = pca.components_[0:1], pca.components_[1:2]
        new_x_range = x_range * d1
        new_y_range = y_range * d2
        print("shape", x_range.shape, d1.shape)
        grid_inputs = torch.FloatTensor(new_x_range + new_y_range).cuda()
        print("grid_inputs", grid_inputs.shape)
    else:
        # grid_inputs = torch.FloatTensor(np.concatenate([x_range, y_range], 1)).cuda()
        # if args.nz > 2:
        #     grid_inputs = torch.cat(
        #         [grid_inputs, z0[:, 2:].expand(grid_inputs.shape[0], -1)], 1
        #     )

    valid_arcs_grid = []
    batch = 1000
    for i in range(0, grid_inputs.shape[0], batch):
        batch_grid_inputs = grid_inputs[i : i + batch, :]
        valid_arcs_grid += batch_grid_inputs
        # valid_arcs_grid += decode_from_latent_space(
        #     batch_grid_inputs, model, 100, max_n, False, data_type
        # )
    print("Evaluating 2D grid points")
    print("Total points: " + str(grid_inputs.shape[0]))
    grid_scores = []
    x, y = [], []
    for i in range(len(valid_arcs_grid)):
        arc = valid_arcs_grid[i]
        if arc is not None:
            score = eval(model, likelihood, test_batch=arc)
            # score = eva.eval(arc)
            x.append(x_range[i, 0])
            y.append(y_range[i, 0])
            grid_scores.append(score)
        else:
            score = 0
        # grid_scores.append(score)
        print(i)
    grid_inputs = grid_inputs.cpu().numpy()
    grid_y = np.array(grid_scores).reshape((n, n))
    save_object((grid_inputs, -grid_y), save_dir + "grid_X_y.dat")
    save_object((x, y, grid_scores), save_dir + "scatter_points.dat")
    if data_type == "BN":
        vmin, vmax = -15000, -11000
    else:
        vmin, vmax = 0.7, 0.76
    ticks = np.linspace(vmin, vmax, 9, dtype=int).tolist()
    cmap = plt.cm.get_cmap("viridis")
    # f = plt.imshow(grid_y, cmap=cmap, interpolation='nearest')
    sc = plt.scatter(x, y, c=grid_scores, cmap=cmap, vmin=vmin, vmax=vmax, s=10)
    plt.colorbar(sc, ticks=ticks)
    plt.savefig(save_dir + "2D_vis.pdf")


def normalize_data(y, mean, std):
    y = (y - mean) / std
    return y


def load_data(root_dir, device=None):
    y_train = np.concatenate(
        pickle.load(open(os.path.join(root_dir, "train_accs.pkl"), "rb")), axis=0
    )

    X_train = np.concatenate(
        pickle.load(open(os.path.join(root_dir, "train_latent_rep.pkl"), "rb")), axis=0
    )

    y_val = np.concatenate(
        pickle.load(open(os.path.join(root_dir, "val_accs.pkl"), "rb")), axis=0
    )
    X_val = np.concatenate(
        pickle.load(open(os.path.join(root_dir, "val_latent_rep.pkl"), "rb")), axis=0
    )

    y_test = np.concatenate(
        pickle.load(open(os.path.join(root_dir, "test_accs.pkl"), "rb")), axis=0
    )
    X_test = np.concatenate(
        pickle.load(open(os.path.join(root_dir, "test_latent_rep.pkl"), "rb")), axis=0
    )

    X_train = torch.from_numpy(X_train).to(device)
    y_train = torch.from_numpy(y_train).to(device).to(dtype=torch.float)
    X_val = torch.from_numpy(X_val).to(device)
    y_val = torch.from_numpy(y_val).to(device).to(dtype=torch.float)
    X_test = torch.from_numpy(X_test).to(device)
    y_test = torch.from_numpy(y_test).to(device).to(dtype=torch.float)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def RMSELoss(y_hat, y):
    return torch.sqrt(torch.mean((y_hat - y) ** 2)).item()


def eval(model, likelihood, test_loader=None, test_batch=None):
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.0])
    with torch.no_grad():
        if test_loader is None:
            preds = model(test_batch)
            means = torch.cat([means, preds.mean.cpu()])
        else:
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
    # np.random.seed(1)
    root_dir = "../dvae/checkpoints/svae_high_lr/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    train_data, val_data, test_data = load_data(root_dir, device)
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

    test_rmses = []
    test_prs = []
    data_type = "BN"  #'BN'NAS

    for exp_no in range(10):
        idx_perm = torch.randperm(X_train.size(0))
        X_train = X_train[idx_perm]
        y_train = y_train[idx_perm]

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
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

        # Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=y_train.size(0), combine_terms=False
        )

        epochs = 100
        best_epoch = 0
        best_model = None
        best_likelihood = None
        best_mae = float("inf")

        for i in range(epochs):
            model.train()
            likelihood.train()
            total_loss = 0.0
            for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
                if minibatch_i >= 5 and data_type == "BN":
                    break
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
            means = eval(model, likelihood, test_loader=val_loader)
            mae = torch.mean(torch.abs(means.to(device=device) - y_val))
            pearsonr = stats.pearsonr(means.cpu().numpy(), y_val.cpu().numpy())[0]
            rmse = RMSELoss(means.to(device=device), y_val)
            print("Val | MAE: {}| pearsonr: {} | rmse: {}".format(mae, pearsonr, rmse))
            if mae < best_mae:
                best_epoch = i
                best_model = model
                best_likelihood = likelihood

        # val
        means = eval(best_model, best_likelihood, test_loader=val_loader)
        mae = torch.mean(torch.abs(means.to(device=device) - y_val))
        pearsonr = stats.pearsonr(means.cpu().numpy(), y_val.cpu().numpy())[0]
        rmse = RMSELoss(means.to(device=device), y_val)
        print(
            "Best Val at {} | MAE: {}| pearsonr: {} | rmse: {}".format(
                best_epoch, mae, pearsonr, rmse
            )
        )
        # test
        means = eval(best_model, best_likelihood, test_loader=test_loader)
        mae = torch.mean(torch.abs(means.to(device=device) - y_test))
        pearsonr = stats.pearsonr(means.cpu().numpy(), y_test.cpu().numpy())[0]
        if not math.isnan(pearsonr):
            test_prs.append(pearsonr)
        rmse = RMSELoss(means.to(device=device), y_test)
        print("Val | MAE: {}| pearsonr: {} | rmse: {}".format(mae, pearsonr, rmse))
        test_rmses.append(rmse)
        visul_2d(best_model, best_likelihood, X_train, data_type='BN', save_dir=root_dir)

    test_rmses = np.array(test_rmses)
    test_prs = np.array(test_prs)
    print("Test: RMSE | Mean: {} | Std: {}".format(test_rmses.mean(), test_rmses.std()))
    print("Test: Pearsonr | Mean: {} | Std: {}".format(test_prs.mean(), test_prs.std()))
    print(root_dir)
    


if __name__ == "__main__":
    main()
