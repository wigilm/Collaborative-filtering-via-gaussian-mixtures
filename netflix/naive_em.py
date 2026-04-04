"""Mixture model using EM"""
from typing import NamedTuple, Tuple
import numpy as np
from common import GaussianMixture
from common import masked_gaussian_pdf

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n = X.shape[0]
    K = mixture.mu.shape[0]
    p = mixture.p
    probs = masked_gaussian_pdf(X, mixture)
    #first i need to calculate the posterior probability, do that for each data point, lets start with one and then loop
    # does Gaussian mixture contain the p's?
    # (K, ) array = each row corresponds to the weight of belonging to cluster j
    weighted_pdf = np.multiply(probs, p)
    weighted_sum = np.sum(weighted_pdf, axis = 1)
    soft_counts = weighted_pdf / weighted_sum.reshape(-1,1)
    log_prob = np.sum(np.log(weighted_sum))
    #converts to column array so broadcasting works
    return soft_counts, log_prob


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    observed = (X != 0)# (n, d)

    # Update means
    weighted_X = post.T @ X  # (K, d)
    denominator = post.T @ observed  # (K, d)
    mu = np.where(denominator > 1e-16, weighted_X / denominator, 0)  # avoid division by 0

    # Update variances (scalar per component)
    var = np.zeros(K)

    for j in range(K):
        diff = (X - mu[j]) * observed  # (n, d)
        squared_diff = diff ** 2  # (n, d)
        weighted_squared = post[:, j].reshape(-1,1) * squared_diff  # (n, d)
        numerator = np.sum(weighted_squared)
        denom = np.sum(post[:, j].reshape(-1,1) * observed)  # (n,1)*(n,d)=(n,d) total observed entries weighted
        var[j] = numerator / denom
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
    while (prev_prob is None or (prob - prev_prob) > 1e-6 * abs(prob)):
        prev_prob = prob
        post, prob = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, prob