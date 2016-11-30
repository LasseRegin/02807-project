
import csv
import numpy as np

from sklearn.externals import joblib

import helpers
import config

# Read tags
tags, tag2idx, tag_count = helpers.read_tags()

# Read words
words, word2idx, word_count = helpers.read_words()

# Get chunks
chunk_reader = helpers.ChunkReader(post_filename=config.paths.TEST_DATA_IDX, chunk_size=config.data.CHUNK_SIZE_TREES)
chunks = [chunk for chunk in chunk_reader]

# Load classifier filenames
classifier_filenames = config.data.load_classifier_filenames()

# Load classifiers
classifiers = [joblib.load(filename) for filename in classifier_filenames]

with open(config.paths.TEST_DATA_IDX, 'r') as f:

    # Count number of true retrieved tags in 'top k'
    true_counts_at_k = {k: 0 for k in range(0, tag_count)}
    total_tag_counts = 0
    for chunk in chunks:

        # Convert to sparse matrix
        X, y_tags = helpers.chunk_to_sparse_mat(chunk, word_count)

        # Predict tag probabilities
        clf_class_probs = []
        for clf in classifiers:
            probs = clf.predict_proba(X)

            # Extract class probabilities
            class_probs = np.asarray([1.0 - prob[:,0] for prob in probs]).T
            clf_class_probs.append(class_probs)

        # Compute mean class probabilities across classifiers
        clf_class_probs = np.asarray(clf_class_probs)
        clf_class_probs = clf_class_probs.mean(axis=0)

        # Sort by highest probability
        sorted_class_indices = clf_class_probs.argsort(axis=1)[:,::-1]

        # Count true retrieved tags
        for i, closest_indices in enumerate(sorted_class_indices):
            true_tags = y_tags[i]
            total_tag_counts += len(true_tags)
            for k in range(0, tag_count):
                for tag in true_tags:
                    if tag in closest_indices[0:k+1]:
                        true_counts_at_k[k] += 1


    # Compute precision at k (P@K)
    precision = {k: true_counts_at_k[k] / total_tag_counts for k in range(0, tag_count)}
    for k, val in precision.items():
        print('P@%d:\t%.4f' % (k+1, val))

    # Save precision at k
    with open(config.paths.PRECISION_AT_K_TREES, 'w') as f:
        writer = csv.writer(f)
        writer.writerow([
            'P@%d' % (k+1) for k in range(0, tag_count)
        ])
        writer.writerow([
            precision[k] for k in range(0, tag_count)
        ])
