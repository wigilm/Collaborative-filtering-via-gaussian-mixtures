"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import common
from common import masked_gaussian_pdf


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    
    """
    n = X.shape[0]
    K = mixture.mu.shape[0]
    p = mixture.p
    f_uj = np.zeros((n,K), dtype=float)
    for k in range(K):
        mu = mixture.mu[k]  # (d,)
        var = mixture.var[k]  # scalar
        p = mixture.p[k]
        for idx, x in enumerate(X):
            observed = x != 0
            data_obs = x[observed]
            mu_obs = mu[observed]
            d_obs = len(data_obs)
            squared_dist = np.dot(data_obs - mu_obs, data_obs - mu_obs)
            log_pdf = np.log(p+1e-16) - d_obs * 0.5 * np.log(2*np.pi*var)-(1/(2*var))*squared_dist
            f_uj[idx, k] = log_pdf
    log_probs = f_uj - logsumexp(f_uj, axis=1).reshape(-1, 1)  # (n, k) - (n, )
    soft_counts = np.exp(log_probs)  # post probability p(j|u)
    proxy_func = np.sum(soft_counts*(f_uj-log_probs))
    return soft_counts, proxy_func


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    observed = (X != 0)  # (n, d)

    # Update means
    weighted_X = post.T @ X  # (K, d)
    denominator = post.T @ observed  # (K, d)
    mu = np.where(denominator >= 1, weighted_X / denominator, mixture.mu)  # avoid division by 0

    # Update variances (scalar per component)
    var = np.zeros(K)

    for j in range(K):
        diff = (X - mu[j]) * observed  # (n, d)
        squared_diff = diff ** 2  # (n, d)
        weighted_squared = post[:, j].reshape(-1,1) * squared_diff  # (n, d)
        numerator = np.sum(weighted_squared)
        denom = np.sum(post[:, j].reshape(-1,1) * observed)  # (n,1)*(n,d)=(n,d) total observed entries weighted
        var[j] = max(numerator / denom, min_variance)
    # Update mixture probabilities
    p = np.sum(post, axis=0) / n  # (K,)

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    prev_prob = None
    prob = None
    while prev_prob is None or (prob - prev_prob) > 1e-6 * abs(prob):
        prev_prob = prob
        post, prob = estep(X, mixture)
        mixture = mstep(X, post, mixture)

    return mixture, post, prob


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    filled_matrix = np.copy(X)
    K = mixture.mu.shape[0]
    f_uj = np.zeros((n, K), dtype=float)
    for k in range(K):
        p = mixture.p[k]
        mu = mixture.mu[k]  # (d,)
        var = mixture.var[k]  # scalar
        for idx, x in enumerate(X):
            observed = x != 0
            data_obs = x[observed]
            mu_obs = mu[observed]
            d_obs = len(data_obs)
            squared_dist = np.dot(data_obs - mu_obs, data_obs - mu_obs)
            log_pdf = np.log(p+1e-16) - d_obs * 0.5 * np.log(2*np.pi*var)-(1/(2*var))*squared_dist
            f_uj[idx, k] = log_pdf
    log_probs = f_uj - logsumexp(f_uj, axis=1).reshape(-1, 1)  # (n, k) - (n, )
    soft_counts = np.exp(log_probs)  # post probability p(j|u)

    for row in range(n):
        for dim in range(d):
            if X[row,dim] == 0:
                filled_matrix[row,dim] = np.dot(soft_counts[row, :], mixture.mu[:, dim]) #(k) (k)
                #post.shape = (n,K)
    return filled_matrix
