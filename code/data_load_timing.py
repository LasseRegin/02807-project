
import os
import time
import math
import multiprocessing
import numpy as np

from helpers import ChunkReader, get_file_size, hash_sentence, encode_tags
import config

#chunk_size = 5 * 1024 ** 2 # 5MB
chunk_size = 20 * 1024 ** 2 # 20MB
#chunk_size = 25 * 1024 ** 2 # 25MB
#chunk_size = 50 * 1024 ** 2 # 50MB
#chunk_size = 100 * 1024 ** 2 # 100MB
#chunk_size = 500 * 1024 ** 2 # 500MB
chunk_reader = ChunkReader(post_filename=config.paths.POST, chunk_size=chunk_size)
chunks = [chunk for chunk in chunk_reader]

K = 20
d = 100
hashing_dim = 100

PROCESS_COUNT = int(os.environ.get('PROCESS_COUNT', multiprocessing.cpu_count()))

def loading_func(chunks, cluster_sums, cluster_counts, mu, lock=None):
    for chunk in chunks:
        with open(config.paths.POST, 'rb') as f:

            # Seek to chunk start bytes
            f.seek(chunk[0])

            # Read end of chunk until end of line end decode it
            chunk_decoded = f.read(chunk[1]).decode('utf-8')

            ## Split in lines (Removing the last newline)
            lines = chunk_decoded.rstrip('\n').split('\n')

            for line in lines:
                # Split in title, body and tags
                lines_splitted = line.split(config.text.delimitter)
                if len(lines_splitted) == 3:
                    title, body, tags = line.split(config.text.delimitter)

                    x = hash_sentence(sentence=title, hashing_dim=hashing_dim)
        # for title, body, tags in chunk_reader.process_chunk(chunk): # Maybe this functions should be away from the object
        #     pass                                                    # since the object needs to be shared on all processes?

def parallel_loading():

    # Initialize shared variable manager
    manager = multiprocessing.Manager()
    lock = multiprocessing.Lock()

    n = math.ceil(len(chunks) / PROCESS_COUNT)
    chunks_split = []
    for i in range(0, len(chunks), n):
        chunks_split.append(chunks[i:i+n])

    cluster_sums = manager.dict({k: np.zeros(d) for k in range(0, K)})
    cluster_counts = manager.dict({k: 0 for k in range(0, K)})
    mu = np.random.rand(d, K)

    # Init processes
    processes = []
    for chunk_list in chunks_split:
        p = multiprocessing.Process(
            target=loading_func,
            #args=(chunk_list, cluster_sums, cluster_counts, lock),
            kwargs={
                'chunks': chunk_list,
                'cluster_sums': cluster_sums,
                'cluster_counts': cluster_counts,
                'mu': mu,
                'lock': lock
            })
        processes.append(p)

    # Start processes
    for p in processes:
        p.start()

    # Wait for processes to finish
    for p in processes:
        p.join()




#N = 10
N = 1

start = time.time()
for _ in range(0, N):
    cluster_sums = {k: np.zeros(d) for k in range(0, K)}
    cluster_counts = {k: 0 for k in range(0, K)}
    mu = np.random.rand(d, K)

    loading_func(chunks=chunks, cluster_sums=cluster_sums, cluster_counts=cluster_counts, mu=mu)
print('Serial loading: %gs' % ((time.time() - start) / N))

start = time.time()
for _ in range(0, N):
    parallel_loading()
print('Parallel loading: %gs' % ((time.time() - start) / N))
