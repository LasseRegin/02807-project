
import os
import re
import csv
import math
from xml.etree import ElementTree as ET

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(FILEPATH, 'data')

def get_tags(tags_dump='/Volumes/Seagate EXP/datasets/stackoverflow-data-dump/stackoverflow/stackoverflow.com-Tags'):
    tag_regex = re.compile(r'(<[^<>]*>)')
    xml_parser = ET.iterparse(tags_dump)
    for i, (_, element) in enumerate(xml_parser):
        if 'TagName' in element.attrib:
            yield {
                'name': element.attrib['TagName'],
                'count': int(element.attrib['Count'])
            }

def get_top_N_tags(N, include_counts=False):
    tags = [tag for tag in get_tags()]
    tags = sorted(tags, key=lambda tag: tag['count'], reverse=True)
    tags = tags[0:N]
    if include_counts:
        return tags
    else:
        return [tag['name'] for tag in tags]


def get_posts(posts_dump='/Volumes/Seagate EXP/datasets/stackoverflow-data-dump/stackoverflow/stackoverflow.com-Posts', max_posts=math.inf):
    tag_regex = re.compile(r'(<[^<>]*>)')
    xml_parser = ET.iterparse(posts_dump)
    for i, (_, element) in enumerate(xml_parser):
        if 'Tags' in element.attrib:
            title = element.attrib.get('Title', '') # Not all have title
            body = element.attrib['Body']
            tags = [tag[1:-1] for tag in re.findall(tag_regex, element.attrib['Tags'])]

            yield {
                'title': title,
                'body': body,
                'tags': tags
            }

            if i > max_posts:   break


if __name__ == '__main__':
    
    # Create data folder
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Get tags
    tags = set(get_top_N_tags(N=20))

    # Save posts to file
    with open(os.path.join(DATA_FOLDER, 'posts-subset.csv'), 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(['title', 'body', 'tags'])
        for post in get_posts():
            if next(filter(tags.__contains__, post['tags']), None) is not None:
                writer.writerow([
                    post['title'],
                    post['body'],
                    ' '.join(post['tags']),
                ])
