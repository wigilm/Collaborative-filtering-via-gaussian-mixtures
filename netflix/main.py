import numpy as np
import kmeans
import common
import naive_em
import em
from common import GaussianMixture
from typing import NamedTuple, Tuple
from rich import print

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
########################################################
#                 IMPLEMENTATION HERE                  #
########################################################

print(f"\n[bold red]For a 2D toy dataset, we will look for the correct implementation of K-means[/bold red]")
cost_vector = np.zeros((4, 5))
for K in range(1, 5):
    for seed in range(5):
        mixture2d, post2d = common.init(X, K, seed)
        mixture2d, post2d, cost = kmeans.run(X, mixture2d, post2d)
        cost_vector[K-1, seed] = cost  # K-1 because K starts at 1
best_seeds = np.argmin(cost_vector, axis=1)  # index of best seed for each K
for K in range(1,5):
    print(f"the best seed for K={K} is seed={best_seeds[K - 1]} with cost = {cost_vector[K - 1, best_seeds[K - 1]]
    }")
    title = f"K-means with K={K}, seed={best_seeds[K-1]}"
    mixture2d, post2d = common.init(X,K,best_seeds[K-1])
    mixture2d, post2d, _ = kmeans.run(X,mixture2d,post2d)
    common.plot(X, mixture2d, post2d, title)

print(f"\n[bold red]Now, in the same 2D toy dataset, we will look for the correct implementation of EM[/bold red]")
cost_vector = np.zeros((4, 5))
for K in range(1, 5):
    for seed in range(5):
        mixture2d, post2d = common.init(X, K, seed)
        mixture2d, post2d, cost = em.run(X, mixture2d, post2d)
        cost_vector[K-1, seed] = cost  # K-1 because K starts at 1
best_seeds = np.argmax(cost_vector, axis=1)  # index of best seed for each K
for K in range(1,5):
    print(f"the best seed for K={K} is seed={best_seeds[K - 1]} with log-likelihood = {cost_vector[K - 1, best_seeds[K - 1]]
    }")
    title = f"E-M with K={K}, seed={best_seeds[K-1]}"
    mixture2d, post2d = common.init(X,K,best_seeds[K-1])
    mixture2d, post2d, _ = em.run(X,mixture2d,post2d)
    common.plot(X, mixture2d, post2d, title)

best_seed, max_loglike = em_algo(netflix_incomplete,12,n_seed=5)
print(f"the max_loglike is = {max_loglike}")
mixture, post = common.init(netflix_incomplete, 12, best_seed)
mixture, post, _ = em.run(netflix_incomplete, mixture, post)
filled_matrix = em.fill_matrix(netflix_incomplete, mixture)
print(common.rmse(filled_matrix,X_gold))
