from mrjob.job import MRJob, MRStep

import numpy as np

import config



my_delimitter = '#MY_CUSTOM_COMMA#'

class MRTest(MRJob):

    def init_get_tags(self):
        # Get tags
        with open(config.paths.TAGS, 'r') as f:
            self.tags = set([tag.rstrip('\n') for tag in f])

        self.tags_count = len(self.tags)
        self.tag2idx = {}
        for i, tag in enumerate(self.tags):
            self.tag2idx[tag] = i

    def mapper(self, _, line):

        #for i, tag in enumerate(self.tags):
        #    yield i, tag

        line_splitted = line.split(my_delimitter)
        title = line_splitted[0]
        body = line_splitted[1]
        tags = line_splitted[2].split(' ')

        target = [0] * self.tags_count
        for tag in tags:
            idx = self.tag2idx.get(tag, -1)
            if idx >= 0:
                target[idx] = 1

        yield "post", (title, target)

    #def reducer(self, key, values):
    #    yield key, sum(values)

    def steps(self):
        return [MRStep(
            mapper_init=self.init_get_tags,
            mapper=self.mapper,
        )]

if __name__ == '__main__':
    MRTest.run()


# class GraphTriangleCounter(MRJob):
#
#     def read_edges(self, _, edge):
#         vertices = edge.split(' ')
#         if len(vertices) == 2:
#             yield int(vertices[0]), int(vertices[1])
#             yield int(vertices[1]), int(vertices[0])
#
#     def get_unique_edges(self, from_vertex, to_vertices):
#         yield from_vertex, list(set(to_vertices))
#
#     def generate_paths(self, from_vertex, to_vertices):
#         for to_vertex in to_vertices:
#             yield sorted((from_vertex, to_vertex)), to_vertices
#
#     def find_common_vertices(self, key, paths):
#         common_vertices = set(next(paths))
#         for path in paths:
#             common_vertices = common_vertices & set(path)
#
#         yield key, list(common_vertices)
#
#     def generate_3_paths(self, key, vertices):
#         for vertex in vertices:
#             yield None, sorted(key + [vertex])
#
#     def get_unique_paths(self, _, paths):
#         unique_paths = set(tuple(path) for path in paths)
#         yield None, 'Number of triangles: %d' % (len(unique_paths))
#
#
#     def steps(self):
#         return [
#             MRStep(mapper=self.read_edges,
#                    reducer=self.get_unique_edges),
#             MRStep(mapper=self.generate_paths,
#                    reducer=self.find_common_vertices),
#             MRStep(mapper=self.generate_3_paths,
#                    reducer=self.get_unique_paths)
#         ]
#
# if __name__ == '__main__':
#     GraphTriangleCounter.run()
