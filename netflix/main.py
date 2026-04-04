import numpy as np
import kmeans
import common
import naive_em
import em
from common import GaussianMixture
from typing import NamedTuple, Tuple

netflix_incomplete = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')

X = np.loadtxt("toy_data.txt")
# TODO: Your code here

def k_mean_algo(X,K,n_seed=5):
    total_cost = np.zeros(5)
    for seed in range(n_seed):
        mixture, post = common.init(X, K, seed)
        cost = kmeans.run(X, mixture, post)[2]
        total_cost[seed] = cost
        best_seed = np.argmin(total_cost)
        min_cost = np.min(total_cost)
    return best_seed, min_cost

def em_algo(X,K,n_seed=5):
    """ finds best seed for certain K and number
    of seeds from 0 to n_seed-1
    """
    # initializes mean vector given X, a seed and K clusters
    total_loglike = np.zeros(5)
    for seed in range(n_seed):
        mixture, post = common.init(X, K, seed)
        log_prob = em.run(X, mixture, post)[2]
        total_loglike[seed] = log_prob
        best_seed = np.argmax(total_loglike)
        max_loglike = np.max(total_loglike)
    return best_seed, max_loglike

def plot_em(X):
    for ka in range(1,5):
        seed, log_like = em_algo(X, ka)
        print(f'the max prob for k={ka} is with seed={seed} with log-like={log_like}')
        mixture, post = common.init(X, ka, seed)
        mixture, post, _ = naive_em.run(X, mixture, post)
        common.plot(X, mixture, post, title=f'cost k={ka}, seed={seed}')

def plot_kmeans(X):
    for ka in range(1,5):
        seed, cost = k_mean_algo(X, ka)
        print(f'the min cost for k={ka} is with seed={seed} with cost={cost}')
        mixture, post = common.init(X, ka, seed)
        mixture, post, _ = kmeans.run(X, mixture, post)
        common.plot(X, mixture, post, title=f'cost k={ka}, seed={seed}')


def find_bestk_em(X, K=np.array([1,12])):
    bic_results = np.zeros(len(K))
    for idx, k in enumerate(K):
        best_seed, log_like = em_algo(X, k) #finds best seed for certain K
        mixture, post = common.init(X, k, best_seed)
        mixture, post, log_likelihood = em.run(X, mixture, post)
        bic_results[idx] = common.bic(X, mixture, log_likelihood)
    best_bic = np.max(bic_results)
    best_K = K[np.argmax(bic_results)]
    return best_K, best_bic

best_seed, max_loglike = em_algo(netflix_incomplete,12,n_seed=5)
print(max_loglike)
mixture, post = common.init(netflix_incomplete, 12, best_seed)
mixture, post, _ = em.run(netflix_incomplete, mixture, post)
filled_matrix = em.fill_matrix(netflix_incomplete, mixture)
print(common.rmse(filled_matrix,X_gold))