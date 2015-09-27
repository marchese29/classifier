from collections import OrderedDict
from math import sqrt

DOCID = 0
CLASS = 1
VECTOR = 2

class KNNClassifier(object):
    def __init__(self, k):
        assert k > 0
        self._k = k

    def _eucl_distance(self, vector1, vector2):
        the_sum = sum(map(lambda x, y: float(x-y)**2, vector1, vector2))
        return sqrt(the_sum)

    def _vector_from_dictionary(self, dictionary):
        od = OrderedDict(sorted(dictionary.items()))
        return [value for key, value in od.iteritems()]

    def train(self, vectors):
        self._vectors = []
        for vector in vectors:
            list_vector = self._vector_from_dictionary(vector[VECTOR])
            for clazz in vector[CLASS]:
                self._vectors.append([vector[DOCID], clazz, list_vector])

    def classify(self, vector):
        # Extract the vector
        list_vector = self._vector_from_dictionary(vector[VECTOR])

        # Find the k nearest neighbors
        data = sorted(self._vectors, key=lambda x: self._eucl_distance(list_vector, x[VECTOR]))
        knn = data[:self._k]

        # Pick a class based on the k nearest neighbors.
        classes = dict()
        for idx in range(self._k):
            if knn[idx][CLASS] in classes:
                classes[knn[idx][CLASS]] += 1
            else:
                classes[knn[idx][CLASS]] = 1
        most_common = max(classes, key=classes.get)
        return [key for key, value in classes.iteritems() if value == most_common]
