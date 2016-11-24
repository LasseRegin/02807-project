
import os
import math
import numpy as np
import multiprocessing

import time

import config
from helpers import ChunkReader, get_file_size, hash_sentence, encode_tags

# Define hashing dimensions
#hashing_dim = 100
hashing_dim = 10
np.random.seed(42)


def train(chunks, cluster_sums, cluster_counts, mu, lock, id):
    for chunk in chunks:
        print('Process %d running on chunk:' % (id+1) , chunk)
        for title, body, tags in chunk_reader.process_chunk(chunk):
            x = hash_sentence(sentence=title, hashing_dim=hashing_dim) # TODO: Use body also
            #y = encode_tags(tags=tags.split(' '))

            # Temp
            dist = (x[:,None] - mu) ** 2
            dist = dist.mean(axis=0)

            # Find closest cluster
            k = dist.argmin()

            # Add to cluster summaries
            cluster_sums[k] += x

            with lock:
                cluster_counts[k] = cluster_counts[k] + 1


if __name__ == '__main__':

    # Read tags
    with open(config.paths.TAGS, 'r') as f:
        tags = set([tag.rstrip('\n') for tag in f])

    tags_count = len(tags)
    tag2idx = {}
    for i, tag in enumerate(tags):
        tag2idx[tag] = i

    # Clusters
    K = tags_count

    # Input space dimensionality
    d = hashing_dim

    # Initialize cluster centers
    mu = np.random.rand(d, K)

    # Determine chunk sizes
    chunk_size = 50 * 1024 ** 2 # 50MB

    # Initialize chunk reader
    chunk_reader = ChunkReader(post_filename=config.paths.POST, chunk_size=chunk_size)
    chunks = [chunk for chunk in chunk_reader]
    chunk_count = len(chunks)

    n = math.ceil(chunk_count / config.algorithm.PROCESS_COUNT)
    chunks_split = []
    for i in range(0, len(chunks), n):
        chunks_split.append(chunks[i:i+n])

    # Initialize shared variable manager
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()

    # Simulate parallel execution
    for i in range(0, 1):

        cluster_sums = manager.dict({k: np.zeros(d) for k in range(0, K)})
        cluster_counts = manager.dict({k: 0 for k in range(0, K)})

        # Init processes
        processes = []
        for i, chunk_list in enumerate(chunks_split):
            p = multiprocessing.Process(target=train, args=(chunk_list, cluster_sums, cluster_counts, mu, lock, i))
            processes.append(p)

        # Start processes
        for p in processes:
            p.start()

        print('Started %d processes' % (len(processes)))

        # Wait for processes to finish
        for p in processes:
            p.join()

        # Update means
        for k in range(0, K):
            counts = cluster_counts[k]
            if counts == 0:  continue

            mu[:,k] = cluster_sums[k] / counts
            # Reset
            cluster_sums[k] = np.zeros(d)
            cluster_counts[k] = 0
