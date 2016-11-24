
import math
import collections
import numpy as np
import multiprocessing
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
chunk_count = len(chunks)

# Split chunks across processes
n = math.ceil(chunk_count / config.algorithm.PROCESS_COUNT)
chunks_split = []
for i in range(0, len(chunks), n):
    chunks_split.append(chunks[i:i+n])

# Initialize shared variable manager
manager = multiprocessing.Manager()
lock = multiprocessing.Lock()

# Define function to run in parallel
def process_chunks(chunks, word_count, K, mu, cluster_sums, cluster_counts, lock):
    for chunk in chunks:

        # Convert to sparse matrix
        X, _ = helpers.chunk_to_sparse_mat(chunk, word_count)

        if X is None:   continue

        # Get closest cluster indices
        max_idx = helpers.sparse_matrix_to_cluster_indices(X, mu)

        mu_subs = collections.defaultdict(list)
        for i, k in enumerate(max_idx):
            mu_subs[k].append(X[i].toarray())

        # Compute sub-means
        for k in range(0, K):
            mu_sub = mu_subs[k]
            if len(mu_sub) == 0:    continue

            with lock:
                cluster_sums[k] = cluster_sums[k] + np.asarray(mu_sub, dtype=np.float32).mean(axis=0)
                cluster_counts[k] += 1


for iteration in range(0, config.algorithm.MAX_ITER):
    start = time.time()

    cluster_sums = manager.dict({k: np.zeros((1, word_count)) for k in range(0, K)})
    cluster_counts = manager.dict({k: 0 for k in range(0, K)})

    # Init processes
    processes = []
    for i, chunk_list in enumerate(chunks_split):
        p = multiprocessing.Process(target=process_chunks, kwargs={
            'chunks': chunk_list,
            'word_count': word_count,
            'K': K,
            'mu': mu,
            'cluster_sums': cluster_sums,
            'cluster_counts': cluster_counts,
            'lock': lock
        })
        processes.append(p)

    # Start processes
    for p in processes:
        p.start()

    #print('Started %d processes' % (len(processes)))

    # Wait for processes to finish
    for p in processes:
        p.join()

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

    if X is None:   continue

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
