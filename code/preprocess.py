
import os
import math
import collections
from nltk.corpus import stopwords

import helpers
import config


if __name__ == '__main__':

    if 'MAX_POSTS' in os.environ:
        MAX_POSTS = int(os.environ['MAX_POSTS'])
    else:
        MAX_POSTS = math.inf

    # Create data folder
    if not os.path.exists(config.paths.DATA_FOLDER):
        os.makedirs(config.paths.DATA_FOLDER)

    # Get tags
    tags = helpers.get_top_N_tags(N=20)

    # Save top tags to file
    with open(config.paths.TAGS, 'w') as f:
        for tag in tags:
            f.write('%s\n' % (tag))

    # Create word counter
    word_counter = collections.Counter()

    # Save posts to file
    config.paths.POST
    text_count = 0
    word_count = 0
    with open(config.paths.POST, 'w') as f:
        for post in helpers.get_posts_filtered(tags, max_posts=MAX_POSTS):
            title = helpers.clean_string(post['title'])
            body = helpers.clean_string(post['body'])
            tags = ' '.join(post['tags'])

            for text in [title, body]:
                for word in text.split():
                    word_counter[word] += 1
                    word_count += 1

            line = config.text.delimitter.join([title, body, tags])

            f.write('%s\n' % (line))
            text_count += 1

    # Save meta data
    config.text.save_meta_data(text_count=text_count)

    # Create dictionary of words to use in Bag of words
    # Only take words occuring atleast 0.1% times and not occuring
    # in more than 50% of the texts
    #min_count = 10
    min_count = text_count // 1000.0
    #min_count = 2 * text_count // 100.0
    max_count = text_count // 2.0

    # Get english stop words
    stop_words = stopwords.words('english')

    with open(config.paths.WORDS, 'w') as f:
        for word, count in word_counter.items():
            if count < min_count:   continue
            if count > max_count:   continue
            if word in stop_words:  continue
            f.write('%s\n' % (word))
