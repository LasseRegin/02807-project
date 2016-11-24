
import helpers
import config
from sklearn.model_selection import train_test_split


# Read tags
tags, tag2idx, tag_count = helpers.read_tags()

# Read words
words, word2idx, word_count = helpers.read_words()

# Get number of texts in data
text_count = config.text.get_text_count()

# Read chunks
chunk_reader = helpers.ChunkReader(post_filename=config.paths.POST, chunk_size=config.data.CHUNK_SIZE_DEBUG) # TODO: Change
all_chunks = [chunk for chunk in chunk_reader]

# Split chunks in training and test
chunks_train, chunks_test = train_test_split(all_chunks, test_size=config.data.TEST_FRACTION)

for chunks, target_filename in [
    (chunks_train, config.paths.TRAIN_DATA_IDX),
    (chunks_test,  config.paths.TEST_DATA_IDX),
]:

    with open(config.paths.POST, 'rb') as f, open(target_filename, 'w') as f_indices:
        for chunk in chunks:

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
                    text = '%s %s' % (title, body)
                    input_vec = []
                    for word in text.split():
                        idx = word2idx.get(word, None)
                        if idx is not None:
                            input_vec.append(idx)

                    target_vec = []
                    for tag in tags.split():
                        idx = tag2idx.get(tag, None)
                        if idx is not None:
                            target_vec.append(idx)

                    input_str  = ' '.join(map(str, input_vec))
                    target_str = ' '.join(map(str, target_vec))

                    f_indices.write('%s,%s\n' % (input_str, target_str))
