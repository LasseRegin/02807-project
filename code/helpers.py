
import os
import re
import math
import numpy as np

from xml.etree import ElementTree as ET
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm as sparse_norm
from sklearn.metrics.pairwise import cosine_similarity

import config

def chunk_to_sparse_mat(chunk, word_count):
    with open(config.paths.TRAIN_DATA_IDX, 'r') as f:
        indptr = [0]
        indices = []
        data = []
        has_data = False
        tags = []
        for i, (input_indices, target_indices) in enumerate(chunk_to_indices(chunk, f)):
            for idx in input_indices:
                indices.append(idx)
                data.append(1)
            indptr.append(len(indices))
            tags.append(list(target_indices))
            has_data = True

        if has_data:
            # NOTE: maybe use float32 datatype
            X = csr_matrix((data, indices, indptr), dtype=np.uint16, shape=(len(indptr) - 1, word_count))

            return X, tags
        else:
            return None, tags

def sparse_matrix_to_cluster_indices(X, mu):
    # Compute cosine similarities
    cos_sims = cosine_similarity(X, mu, dense_output=True)
    max_idx = cos_sims.argmax(axis=1)

    return max_idx

def sparse_matrix_to_sorted_cluster_indices(X, mu):
    # Compute cosine similarities
    cos_sims = cosine_similarity(X, mu, dense_output=True)
    sorted_idx = cos_sims.argsort(axis=1)[:,::-1]

    return sorted_idx


def chunk_to_indices(chunk, f):
    # Seek to chunk start bytes
    f.seek(chunk[0])

    # Read end of chunk until end of line
    chunk_decoded = f.read(chunk[1])

    ## Split in lines (Removing the last newline)
    lines = chunk_decoded.rstrip('\n').split('\n')

    for line in lines:
        line_splitted = line.split(',')
        if len(line_splitted) == 2:
            input_indices  = map(int, line_splitted[0].split(' '))
            target_indices = map(int, line_splitted[1].split(' '))
            yield input_indices, target_indices


def get_file_size(filename):
    st = os.stat(filename)
    return st.st_size

def hash_word(word, hashing_dim):
    return sum(ord(a) for a in word) % hashing_dim

def hash_sentence(sentence, hashing_dim):
    vec = np.zeros(hashing_dim).astype('uint32')
    for word in sentence.split(' '):
        vec[hash_word(word, hashing_dim)] += 1
    return vec

def encode_tags(tags, tags_count):
    target = np.zeros(tags_count)
    for tag in tags:
        idx = tag2idx.get(tag, -1)
        if idx > -1:
            target[idx] = 1
    return target.astype('uint8')


def read_tags():
    with open(config.paths.TAGS, 'r') as f:
        tags = set([tag.rstrip('\n') for tag in f])
    tags = list(sorted(tags))

    tag_count = len(tags)
    tag2idx = {}
    for i, tag in enumerate(tags):
        tag2idx[tag] = i

    return tags, tag2idx, tag_count

def read_words():
    with open(config.paths.WORDS, 'r') as f:
        words = set([word.rstrip('\n') for word in f])
    words = list(sorted(words))

    word_count = len(words)
    word2idx = {}
    for i, word in enumerate(words):
        word2idx[word] = i

    return words, word2idx, word_count


class ChunkReader:
    def __init__(self, post_filename, chunk_size=1024*1024):
        self.post_filename = post_filename
        self.chunk_size = chunk_size

    def __iter__(self):
        with open(self.post_filename, 'rb') as f:
            while True:
                start = f.tell()
                f.seek(self.chunk_size, 1)
                s = f.readline()
                yield start, f.tell() - start
                if not s:   break

    def process_chunk(self, chunk):
        with open(self.post_filename, 'rb') as f:

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
                    yield line.split(config.text.delimitter)


# Precompile regular expressions
reg_links  = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
re_digits  = re.compile(r'\b\d+\b')
re_spaces  = re.compile(r'\s{2,}')

reg_symbols = re.compile(r'[^A-Za-z0-9(),!?\'\`]')
reg_symb_1 = re.compile(r',')
reg_symb_2 = re.compile(r'!')
reg_symb_3 = re.compile(r'\(')
reg_symb_4 = re.compile(r'\)')
reg_symb_5 = re.compile(r'\?')
reg_symb_6 = re.compile(r'\'')

reg_suf_1 = re.compile(r'\'s')
reg_suf_2 = re.compile(r'\'ve')
reg_suf_3 = re.compile(r'n\'t')
reg_suf_4 = re.compile(r'\'re')
reg_suf_5 = re.compile(r'\'d')
reg_suf_6 = re.compile(r'\'ll')

def clean_string(text):
    # Replace links with link identifier
    text = reg_links.sub('<link>', text)

    # Remove certain symbols
    text = reg_symbols.sub(' ', text)

    # Remove suffix from words
    # TODO: Maybe replace english suffix with real word
    text = reg_suf_1.sub(' ', text)
    text = reg_suf_2.sub(' ', text)
    text = reg_suf_3.sub(' ', text)
    text = reg_suf_4.sub(' ', text)
    text = reg_suf_5.sub(' ', text)
    text = reg_suf_6.sub(' ', text)

    # Remove "'" from string
    text = reg_symb_6.sub('', text)

    # Replace breaks with spaces
    text = text.replace('<br />', ' ')
    text = text.replace('\r\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')

    # Pad symbols with spaces on both sides
    text = reg_symb_1.sub(' , ', text)
    text = reg_symb_2.sub(' ! ', text)
    text = reg_symb_3.sub(' ( ', text)
    text = reg_symb_4.sub(' ) ', text)
    text = reg_symb_5.sub(' ? ', text)

    # Replace digits with 'DIGIT'
    text = re_digits.sub('<DIGIT>', text)

    # Remove double whitespaces
    text = re_spaces.sub(' ', text)
    text = text.strip()

    # Convert to lowercase
    text = text.lower()

    return text


def get_tags():
    xml_parser = ET.iterparse(config.paths.TAGS_DUMP)
    for i, (_, element) in enumerate(xml_parser):
        if 'TagName' in element.attrib:
            yield {
                'name': element.attrib['TagName'],
                'count': int(element.attrib['Count'])
            }
        element.clear()

def get_top_N_tags(N, include_counts=False):
    tags = [tag for tag in get_tags()]
    tags = sorted(tags, key=lambda tag: tag['count'], reverse=True)
    tags = tags[0:N]
    if include_counts:
        return tags
    else:
        return [tag['name'] for tag in tags]


def get_posts(max_posts=math.inf):
    tag_regex = re.compile(r'(<[^<>]*>)')
    xml_parser = ET.iterparse(config.paths.POST_DUMP)
    for i, (_, element) in enumerate(xml_parser):
        if 'Tags' in element.attrib:
            title = element.attrib.get('Title', '') # Not all have title
            body = element.attrib['Body']
            tags = [tag[1:-1] for tag in tag_regex.findall(element.attrib['Tags'])]

            yield {
                'title': title,
                'body': body,
                'tags': tags
            }

            if i > max_posts:   break
        element.clear()


def get_posts_filtered(tags, **kwargs):
    tags = set(tags)
    for post in get_posts(**kwargs):
        if next(filter(tags.__contains__, post['tags']), None) is not None:
            yield post



if __name__ == '__main__':

    chunk_reader = ChunkReader(post_filename=config.paths.POST, chunk_size=1024)
    for chunk in chunk_reader:
        print(chunk)
