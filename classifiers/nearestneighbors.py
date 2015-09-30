from collections import OrderedDict
from math import sqrt

import numpy as np
from scipy.sparse import dok_matrix

DOCID = 0
CLASS = 1
VECTOR = 2

class KNNClassifier(object):
    def __init__(self, k):
        assert k > 0
        self._k = k

    def train(self, vectors):
        self._vectors = []
        for vector in vectors:
            for topic in vector[CLASS]:
                self._vectors.append([vector[DOCID], topic, vector[VECTOR]])

    def classify(self, vector):
        # Find the k nearest neighbors
        result = []
        # np.linalg.norm is just a neat way of finding euclidean distance, this is a DRAMATIC
        # performance improvement over simple python lists.
        data = sorted(self._vectors, key=lambda x: np.linalg.norm(vector[VECTOR] - x[VECTOR]))
        knn = data[:self._k]

        # Pick a class based on the k nearest neighbors.
        classes = dict()
        for idx in range(self._k):
            if knn[idx][CLASS] in classes:
                classes[knn[idx][CLASS]] += 1
            else:
                classes[knn[idx][CLASS]] = 1

        # Find the most common element, use it as the class for this instance.
        max_count = 0
        most_common = ''
        for key, value in classes.iteritems():
            if value > max_count:
                max_count = value
                most_common = key
        return most_common
