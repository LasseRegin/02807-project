import os
import pickle
import multiprocessing

FILEPATH = os.path.dirname(os.path.abspath(__file__))
class paths:
    DATA_FOLDER = os.path.join(FILEPATH, 'data')

    # Posts
    POST = os.path.join(DATA_FOLDER, 'posts.csv')
    POST_DUMP = '/Volumes/Seagate EXP/datasets/stackoverflow-data-dump/stackoverflow/stackoverflow.com-Posts'

    # Tags
    TAGS = os.path.join(DATA_FOLDER, 'tags.csv')
    TAGS_DUMP = '/Volumes/Seagate EXP/datasets/stackoverflow-data-dump/stackoverflow/stackoverflow.com-Tags'

    # Words
    WORDS = os.path.join(DATA_FOLDER, 'words.csv')

    # Meta data
    META = os.path.join(DATA_FOLDER, 'meta.pkl')

    # Input/target indices
    TRAIN_DATA_IDX = os.path.join(DATA_FOLDER, 'train-data-indices.csv')
    TEST_DATA_IDX  = os.path.join(DATA_FOLDER, 'test-data-indices.csv')

    # Mean numpy array
    MU = os.path.join(DATA_FOLDER, 'means.dat')

    # Cluster tags dict
    CLUSTER_TAGS = os.path.join(DATA_FOLDER, 'cluster-tags.pkl')

    # Evaluations
    PRECISION_AT_K = os.path.join(DATA_FOLDER, 'precision.csv')

    # # Memory-mapped files
    # MEM_MAP_INPUT  = os.path.join(DATA_FOLDER, 'mem_map_input.dat')
    # MEM_MAP_TARGET = os.path.join(DATA_FOLDER, 'mam_map_target.dat')


class data:
    TEST_FRACTION = 0.33

    CHUNK_SIZE = 20 * 1024 ** 2 # 20MB
    CHUNK_SIZE_DEBUG = 1024 ** 2  # 1MB

    @classmethod
    def save_cluster_tags(cls, cluster_tags):
        with open(paths.CLUSTER_TAGS, 'wb') as f:
            pickle.dump(cluster_tags, f)

    @classmethod
    def load_cluster_tags(cls):
        with open(paths.CLUSTER_TAGS, 'rb') as f:
            return pickle.load(f)


class algorithm:
    # Convergence criteria
    MAX_ITER = 1000
    EPSILON = 1e-10

    # Number of processes to use in parallel
    # TODO: Maybe use 2 * cpu_count (Hyperthreading)
    PROCESS_COUNT = int(os.environ.get('PROCESS_COUNT', multiprocessing.cpu_count()))




class text:
    delimitter = '#MY_CUSTOM_COMMA#'

    @classmethod
    def get_text_count(cls):
        meta_data = cls.load_meta_data()
        return meta_data['text_count']

    @classmethod
    def save_meta_data(cls, text_count):
        meta_data = {
            'text_count': text_count
        }
        with open(paths.META, 'wb') as f:
            pickle.dump(meta_data, f)

    @classmethod
    def load_meta_data(cls):
        with open(paths.META, 'rb') as f:
            return pickle.load(f)
