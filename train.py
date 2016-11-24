
import collections
from scipy.sparse import csr_matrix
import numpy as np
import time

import helpers
import config

#np.random.seed(42)

# Read tags
tags, tag2idx, tag_count = helpers.read_tags()

# Read words
words, word2idx, word_count = helpers.read_words()

# Clusters
K = tag_count

# Initialize cluster centers
mu = np.random.rand(K, word_count)

# Get chunks
chunk_reader = helpers.ChunkReader(post_filename=config.paths.TRAIN_DATA_IDX, chunk_size=config.data.CHUNK_SIZE_DEBUG) # TODO: Change
chunks = [chunk for chunk in chunk_reader]


#with open(config.paths.TRAIN_DATA_IDX, 'r') as f:
for iteration in range(0, config.algorithm.MAX_ITER):
    start = time.time()

    cluster_sums   = {k: np.zeros((1, word_count)) for k in range(0, K)}
    cluster_counts = {k: 0 for k in range(0, K)}

    for chunk in chunks:

        # Convert to sparse matrix
        X, _ = helpers.chunk_to_sparse_mat(chunk, word_count)

        # Get closest cluster indices
        max_idx = helpers.sparse_matrix_to_cluster_indices(X, mu)

        mu_subs = collections.defaultdict(list)
        for i, k in enumerate(max_idx):
            mu_subs[k].append(X[i].toarray())

        # Compute sub-means
        for k in range(0, K):
            mu_sub = mu_subs[k]
            if len(mu_sub) == 0:    continue
            cluster_sums[k] += np.asarray(mu_sub, dtype=np.float32).mean(axis=0)
            cluster_counts[k] += 1

    # Save old means
    mu_old = np.array(mu, copy=True)

    # Update means
    for k in range(0, K):
        count = cluster_counts[k]
        if count == 0:  continue
        mu[k] = cluster_sums[k] / cluster_counts[k]

    # Check convergence criteria
    mu_norm = np.linalg.norm(mu - mu_old)

    print('Iteration took: %.4fs' % (time.time() - start))

    if mu_norm < config.algorithm.EPSILON:
        print('Converged after %d iterations' % (iteration+1))
        break


# Determine cluster tags
cluster_tag_counts = {k: {tag: 0 for tag in range(0, K)} for k in range(0, K)}
for chunk in chunks:

    # Convert to sparse matrix
    X, tags = helpers.chunk_to_sparse_mat(chunk, word_count)

    # Get closest cluster indices
    max_idx = helpers.sparse_matrix_to_cluster_indices(X, mu)

    # Count cluster tags
    for i, k in enumerate(max_idx):
        for tag_idx in tags[i]:
            cluster_tag_counts[k][tag_idx] += 1

# Assign tags to clusters
tags_labelled = []
cluster2tag = {}
for k, tag_counts in cluster_tag_counts.items():
    tag_counts_sorted = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    for tag, count in tag_counts_sorted:
        if tag not in tags_labelled:
            cluster2tag[k] = tag
            tags_labelled.append(tag)
            break

# Save cluster tags dict
config.data.save_cluster_tags(cluster_tags=cluster2tag)

# Save means
with open(config.paths.MU, 'wb') as f:
    np.save(f, mu)
