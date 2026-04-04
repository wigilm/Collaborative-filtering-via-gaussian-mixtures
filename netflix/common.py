"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Arc


class GaussianMixture(NamedTuple):
    """Tuple holding a gaussian mixture"""
    mu: np.ndarray  # (K, d) array - each row corresponds to a gaussian component mean
    var: np.ndarray  # (K, ) array - each row corresponds to the variance of a component
    p: np.ndarray  # (K, ) array = each row corresponds to the weight of a component


def init(X: np.ndarray, K: int,
         seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assingments

    Args:
        X: (n, d) array holding the data
        K: number of components
        seed: random seed

    Returns:
        mixture: the initialized gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    """
    np.random.seed(seed)
    n, _ = X.shape
    p = np.ones(K) / K

    # select K random points as initial means
    mu = X[np.random.choice(n, K, replace=False)]
    var = np.zeros(K)
    # Compute variance
    for j in range(K):
        var[j] = ((X - mu[j])**2).mean()

    mixture = GaussianMixture(mu, var, p)
    post = np.ones((n, K)) / K

    return mixture, post


def plot(X: np.ndarray, mixture: GaussianMixture, post: np.ndarray,
         title: str):
    """Plots the mixture model for 2D data"""
    _, K = post.shape

    percent = post / post.sum(axis=1).reshape(-1, 1)
    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-20, 20))
    ax.set_ylim((-20, 20))
    r = 0.25
    color = ["r", "b", "k", "y", "m", "c"]
    for i, point in enumerate(X):
        theta = 0
        for j in range(K):
            offset = percent[i, j] * 360
            arc = Arc(point,
                      r,
                      r,
                      angle=0,
                      theta1=theta,
                      theta2=theta + offset,
                      edgecolor=color[j])
            ax.add_patch(arc)
            theta += offset
    for j in range(K):
        mu = mixture.mu[j]
        sigma = np.sqrt(mixture.var[j])
        circle = Circle(mu, sigma, color=color[j], fill=False)
        ax.add_patch(circle)
        legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
            mu[0], mu[1], sigma)
        ax.text(mu[0], mu[1], legend)
    plt.axis('equal')
    plt.show()


def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def bic(X: np.ndarray, mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """Computes the Bayesian Information Criterion for a
    mixture of gaussians

    Args:
        X: (n, d) array holding the data
        mixture: a mixture of spherical gaussian
        log_likelihood: the log-likelihood of the data

    Returns:
        float: the BIC for this mixture
    """
    n = X.shape[0]
    k, d = mixture.mu.shape
    p = k*d+ k*2 - 1 #due to restriction sum_k p_k = 1
    bic = log_likelihood - 0.5 * p * np.log(n)
    return bic


def multivariate_Gaussianpdf(data, mixture: GaussianMixture):
    """
    prints the pdf of a multivariate gaussian given data x and the parameters mixture
    x is a n*d dimensional vector
    K is the number of clusters
    mixture refers to the class GaussianMixture which contains Gaussian parameters

    returns n*K in which each row is the pdf of each vector for each cluster k
    """

    n, d = data.shape
    K = mixture.mu.shape[0]

    result = np.zeros((n, K), dtype=float)

    for k in range(K):
        mu = mixture.mu[k]  # (d,)
        var = mixture.var[k]  # scalar
        cov_det = (2 * np.pi * var) ** (d / 2)

        for i in range(n):
            diff = data[i] - mu  # (d,)
            squared_dist = np.dot(diff, diff)
            exponent = -0.5 * squared_dist / var
            result[i, k] = (1 / cov_det) * np.exp(exponent)

    return result  # shape: (n, K)


def masked_gaussian_pdf(data, mixture: GaussianMixture):
    """
    this function will calculate the correct Gaussian pdf
    treating missing entries as 0's, no redundancies or
    misscalculations.

    returns -> probs: a n,k matrix, rows indicate
    the data point and columns each score for one of the k pdfs.s
    """
    n = data.shape[0]
    K = mixture.mu.shape[0]
    probs = np.zeros((n,K),dtype=float)
    for k in range(K):
        mu = mixture.mu[k]  # (d,)
        var = mixture.var[k]  # scalar
        for idx, x in enumerate(data):
            observed = x != 0
            data_obs = x[observed]
            mu_obs = mu[observed]
            d_obs = len(data_obs)
            squared_dist = np.dot(data_obs - mu_obs, data_obs - mu_obs)
            normalizer = (2 * np.pi * var) ** (d_obs / 2)
            exponent = -0.5 * squared_dist / var
            pdf = (1 / normalizer) * np.exp(exponent)
            probs[idx, k] = pdf
    return probs
