import os

os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["THEANO_FLAGS"] = "device=cpu"
import pickle
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns
import matplotlib.pyplot as plt
import pmlearn
from pmlearn.gaussian_process import (
    SparseGaussianProcessRegressor,
    GaussianProcessRegressor,
)


def normalize_data(y, mean, std):
    y = (y - mean) / std
    return y


def main():
    np.random.seed(12345)
    rc = {
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "axes.labelsize": 20,
        "font.size": 20,
        "legend.fontsize": 12.0,
        "axes.titlesize": 10,
        "figure.figsize": [12, 6],
    }
    sns.set(rc=rc)

    # n = 150  # The number of data points
    # X = np.linspace(start=0, stop=10, num=n)[
    #     :, None
    # ]  # The inputs to the GP, they must be arranged as a column vector

    y_train = np.concatenate(pickle.load(open(f"../train_accs.pkl", "rb")), axis=0)
    print(len(y_train))
    z_train = np.concatenate(
        pickle.load(open(f"../train_latent_rep.pkl", "rb")), axis=0
    )
    print(len(z_train))

    y_val = np.concatenate(pickle.load(open(f"../val_accs.pkl", "rb")), axis=0)
    print(len(y_val))
    z_val = np.concatenate(pickle.load(open(f"../val_latent_rep.pkl", "rb")), axis=0)
    print(len(z_val))

    y_test = np.concatenate(pickle.load(open(f"../test_accs.pkl", "rb")), axis=0)
    print(len(y_test))
    z_test = np.concatenate(pickle.load(open(f"../test_latent_rep.pkl", "rb")), axis=0)
    print(len(z_test))

    y_train_mean = np.mean(y_train)
    y_train_std = np.std(y_train)

    print(y_train_mean, y_train_std)
    y_train = normalize_data(y_train, y_train_mean, y_train_std)
    y_val = normalize_data(y_val, y_train_mean, y_train_std)
    y_test = normalize_data(y_test, y_train_mean, y_train_std)

    # The observed data is the latent function plus a small amount of Gaussian distributed noise
    # The standard deviation of the noise is `sigma`
    # f_true = []
    # noise_variance_true = 2.0
    # y = f_true + noise_variance_true * np.random.randn(n)

    model = GaussianProcessRegressor()
    model.fit(z_train, y_train)
    model.plot_elbo()

    pm.traceplot(model.trace)

    y_pred_val = model.predict(z_val)
    print("val:", model.score(z_val, y_val))
    print("test:", model.score(z_test, y_test))


if __name__ == "__main__":
    main()
