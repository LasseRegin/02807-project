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
    PRECISION_AT_K_TREES = os.path.join(DATA_FOLDER, 'precision-trees.csv')

    # Classifiers folder
    MODELS_FOLDER = os.path.join(FILEPATH, 'models')

    # Classifier filename
    CLASSIFIERS = os.path.join(DATA_FOLDER, 'classifiers.csv')


class data:
    TEST_FRACTION = 0.33

    #CHUNK_SIZE = 20 * 1024 ** 2 # 20MB
    #CHUNK_SIZE = 10 * 1024 ** 2 # 10MB
    #CHUNK_SIZE = 5 * 1024 ** 2 # 5MB
    #CHUNK_SIZE = 2 * 1024 ** 2 # 2MB
    CHUNK_SIZE = 1 * 1024 ** 2 # 1MB
    #CHUNK_SIZE_TREES = 1 * 1024 ** 2 # 1MB
    CHUNK_SIZE_TREES = 2 * 1024 ** 2 # 2MB
    #CHUNK_SIZE_TREES = 5 * 1024 ** 2 # 2MB
    #CHUNK_SIZE_TREES = 10 * 1024 ** 2 # 2MB

    @classmethod
    def save_cluster_tags(cls, cluster_tags):
        with open(paths.CLUSTER_TAGS, 'wb') as f:
            pickle.dump(cluster_tags, f)

    @classmethod
    def load_cluster_tags(cls):
        with open(paths.CLUSTER_TAGS, 'rb') as f:
            return pickle.load(f)

    @classmethod
    def save_classifier_filenames(cls, classifier_filenames):
        with open(paths.CLASSIFIERS, 'w') as f:
            for filename in classifier_filenames:
                f.write('%s\n' % (filename))


    @classmethod
    def load_classifier_filenames(cls):
        with open(paths.CLASSIFIERS, 'r') as f:
            filenames = [filename.rstrip('\n') for filename in f]
        return filenames


class algorithm:
    # Convergence criteria
    MAX_ITER = 1000
    EPSILON = 1e-10

    # Number of processes to use in parallel
    # TODO: Maybe use 2 * cpu_count (Hyperthreading)
    PROCESS_COUNT = int(os.environ.get('PROCESS_COUNT', multiprocessing.cpu_count()))
    #PROCESS_COUNT = int(os.environ.get('PROCESS_COUNT', multiprocessing.cpu_count() * 2))




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
