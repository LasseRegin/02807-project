
import csv
import numpy as np

import helpers
import config

# Read tags
tags, tag2idx, tag_count = helpers.read_tags()

# Read words
words, word2idx, word_count = helpers.read_words()

# Look at top 5 predictions
top_k = 5

# Load means
with open(config.paths.MU, 'rb') as f:
    mu = np.load(f)

# Get chunks
chunk_reader = helpers.ChunkReader(post_filename=config.paths.TEST_DATA_IDX, chunk_size=config.data.CHUNK_SIZE_DEBUG) # TODO: Change
chunks = [chunk for chunk in chunk_reader]

# Load cluster tags dict
cluster2tag = config.data.load_cluster_tags()

with open(config.paths.TEST_DATA_IDX, 'r') as f:

    # Count number of true retrieved tags in 'top k'
    true_counts_at_k = {k: 0 for k in range(0, tag_count)}
    total_tag_counts = 0
    for chunk in chunks:

        # Convert to sparse matrix
        X, y_tags = helpers.chunk_to_sparse_mat(chunk, word_count)

        # Get closest cluster indices
        sorted_idx = helpers.sparse_matrix_to_sorted_cluster_indices(X, mu)

        # Count true retrieved tags
        for i, closest_indices in enumerate(sorted_idx):
            true_tags = [cluster2tag[idx] for idx in y_tags[i]]
            total_tag_counts += len(true_tags)
            for k in range(0, tag_count):
                tag_predictions = [cluster2tag[cluster] for cluster in closest_indices[0:k+1]]
                for tag in true_tags:
                    if tag in tag_predictions:
                        true_counts_at_k[k] += 1

    # Compute precision at k (P@K)
    precision = {k: true_counts_at_k[k] / total_tag_counts for k in range(0, tag_count)}
    for k, val in precision.items():
        print('P@%d:\t%.4f' % (k+1, val))

    # Save precision at k
    with open(config.paths.PRECISION_AT_K, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'P@%d' % (k+1) for k in range(0, tag_count)
        ])
        writer.writerow([
            precision[k] for k in range(0, tag_count)
        ])
