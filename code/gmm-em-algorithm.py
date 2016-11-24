

from scipy.stats import multivariate_normal
from sklearn import datasets
import numpy as np

# Load data and permute
iris = datasets.load_iris()
#np.random.seed(1)
X = iris.data
Y = iris.target
n, d = X.shape
idx = np.random.permutation(n)
X = X[idx,:]
Y = Y[idx]

# Normalize input
X = (X - X.mean()) / X.std()

# Number of clusters
K = 3

# Initialize means a convariance matrices
mu    = np.random.rand(d, K)
sigma = np.random.rand(d, d, K)
for k in range(0, K):
    sigma[:,:,k] = np.dot(sigma[:,:,k], sigma[:,:,k].T)

# Mixing coefficients (TODO: Maybe use class fraction?)
pi = np.random.rand(K)
pi /= pi.sum()

max_iter = 50
log_like_window = 5

# Initialize arrays
pi_N_products = np.zeros((n, K))
N_vals = np.zeros((n, K))
gamma = np.zeros((n, K))
log_likelihood = np.zeros(max_iter)


alpha = 0.1
epsilon = 1e-10 # Convergence criteria

# Perform EM-algorithm
for i in range(0, max_iter):

    #
    # E-step
    #

    # Compute pi_k*N_vals_k products
    for k in range(0, K):
        N_vals[:,k] = multivariate_normal.pdf(X, mean=mu[:,k], cov=sigma[:,:,k] + np.eye(d) * alpha)
        pi_N_products[:,k] = pi[k] * N_vals[:,k]

    # Compute product sum over all clusters (inner sum of log-likelihood)
    pi_N_products_sum = pi_N_products.sum(axis=1)
    pi_N_products_sum_log = np.log(pi_N_products_sum)
    log_likelihood[i] = pi_N_products_sum_log.sum()

    if i >= log_like_window:
        likelihoods = log_likelihood[i+1-log_like_window:i+1]

        if likelihoods.std() > epsilon:
            print('Converged after %d iterations' % (i))
            break


    # TODO: Add convergence criteria

    # Compute responsibilities
    for k in range(0, K):
        gamma[:,k] = pi_N_products[:,k] / pi_N_products_sum

    # Assign clusters
    cluster_assignments = N_vals.argmax(axis=1)
    cluster_counts = np.bincount(cluster_assignments)
    cluster_counts_length = len(cluster_counts)

    N_k = gamma.sum(axis=0)
    for k in range(0, K):
        if N_k[k] == 0: continue
        frac = 1.0 / N_k[k]

        # Update means
        gamma_X_product = gamma[:,k,None] * X
        mu[:,k] = frac * gamma_X_product.sum(axis=0)

        # Update covariances
        X_norm = X - mu[:,k]
        sigma_sum = np.zeros((d,d))
        for i in range(0, n):
            dot_prod = np.outer(X_norm[i].T, X_norm[i])
            sigma_sum += gamma[i,k] * dot_prod
        sigma[:,:,k] = frac * sigma_sum

        # Update mixing coefficients
        pi[k] = N_k[k] / n


# Compute probabilities
N_vals = np.zeros((n, K))
for k in range(0, K):
    N_vals[:,k] = multivariate_normal.pdf(X, mean=mu[:,k], cov=sigma[:,:,k] + np.eye(d) * alpha)

# Assign to cluster
cluster_assignments = N_vals.argmax(axis=1)

for k in range(0, K):
    print('Cluster %d' % (k))
    print(Y[cluster_assignments == k])
