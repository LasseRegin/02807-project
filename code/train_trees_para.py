
import os
import math
import collections
import numpy as np
import multiprocessing
from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import helpers
import config

# Create models folder
if not os.path.exists(config.paths.MODELS_FOLDER):
    os.makedirs(config.paths.MODELS_FOLDER)

# Read tags
tags, tag2idx, tag_count = helpers.read_tags()

# Read words
words, word2idx, word_count = helpers.read_words()

# Get chunks
chunk_reader = helpers.ChunkReader(post_filename=config.paths.TRAIN_DATA_IDX, chunk_size=config.data.CHUNK_SIZE_TREES) # TODO: Change
chunks = [chunk for chunk in chunk_reader]
chunk_count = len(chunks)

# Filesize total
bytes_total = sum(chunks[-1])

# Split chunks across processes
n = math.ceil(chunk_count / config.algorithm.PROCESS_COUNT)
chunks_split = []
for i in range(0, len(chunks), n):
    chunks_split.append(chunks[i:i+n])

# Initialize shared variable manager
manager = multiprocessing.Manager()
lock = multiprocessing.Lock()

# Define function to run in parallel
def process_chunks(chunks, word_count, tag_count, clf_folder, classifier_filenames, bytes_processed, bytes_total, lock):
    for chunk in chunks:

        # Convert to sparse matrix
        X, target_indices = helpers.chunk_to_sparse_mat(chunk, word_count)

        if X is None:   continue

        # Create target vector from target indices
        Y = np.zeros((len(target_indices), tag_count))
        for i, indices in enumerate(target_indices):
            Y[i,indices] = 1

        # Train decision tree
        clf = DecisionTreeClassifier(
            splitter='best',
            max_features='auto',
            max_depth=None,
        )

        # Fit data
        clf.fit(X.toarray(), Y)

        # Save trained classifier
        classifier_filename = os.path.join(clf_folder, 'clf-%s-%s.pkl' % chunk)
        joblib.dump(clf, classifier_filename)

        # Add classifier name to file
        with lock:
            classifier_filenames.append(classifier_filename)
            bytes_processed.value += chunk[1]
            print('Processed: %d/%d' % (bytes_processed.value, bytes_total))


classifier_filenames = manager.list([])
bytes_processed = manager.Value('i', 0)

# Init processes
processes = []
for i, chunk_list in enumerate(chunks_split):
    p = multiprocessing.Process(target=process_chunks, kwargs={
        'chunks': chunk_list,
        'word_count': word_count,
        'tag_count': tag_count,
        'clf_folder': config.paths.MODELS_FOLDER,
        'classifier_filenames': classifier_filenames,
        'bytes_processed': bytes_processed,
        'bytes_total': bytes_total,
        'lock': lock
    })
    processes.append(p)

# Start processes
for p in processes:
    p.start()

# Wait for processes to finish
for p in processes:
    p.join()

# Save classifier filenames to file
config.data.save_classifier_filenames(classifier_filenames)
