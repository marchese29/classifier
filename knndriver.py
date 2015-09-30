from collections import OrderedDict
import pickle
import sys
import time
import traceback

import numpy as np

from classifiers.nearestneighbors import KNNClassifier
from util.crossValidation import crossValidate

def vector_from_dictionary(dictionary, words=[]):
        od = OrderedDict(sorted(dictionary.items()))
        return [value for key, value in od.iteritems() if len(words) == 0 or key in words]

def test_classifier(training_data, test_data, classifier):
    print '\tTraining the classifier (indexing input)'
    start_time = time.time()
    classifier.train(training_data)
    elapsed = time.time() - start_time
    print '\tTraining time: %s seconds' % elapsed
    print '\tTraining time per vector: %s seconds' % (elapsed / len(training_data))

    correct = 0
    print '\tValidating against test data.'
    start_time = time.time()
    for idx, vector in enumerate(test_data):
        result = classifier.classify(vector)
        if result in vector[1]:
            correct += 1
    elapsed = time.time() - start_time
    print '\tTime taken: %s seconds' % elapsed
    print '\tAverage classification time: %s seconds' % (elapsed / len(test_data))

    accuracy = float(correct) / float(len(test_data))
    print '\tAccuracy: %f' % accuracy
    return accuracy

def main():
    print 'Loading the feature vectors...this may take a little while.'
    with open('data/reducedFeatureVectorMatrix1000.pkl', 'r') as f:
        data = pickle.load(f)

    print 'Processing the feature vectors.'
    with open('data/words.pickle', 'r') as words_file:
        words = pickle.load(words_file)
    vectors = []
    for vector in data:
        np_vector = np.array(vector_from_dictionary(vector[2], words=words))
        # Use this opportunity to eliminate null vectors.
        if np_vector.any():
            vectors.append([vector[0], vector[1], np_vector])

    print 'Partitioning the data set for cross-validation.'
    partitions = crossValidate(vectors)

    print 'Testing the KNN Classifier with K=11'
    acc_sum = 0.0
    for idx, partition in enumerate(partitions):
        print 'Testing partition #%d' % (idx + 1)
        acc_sum += test_classifier(partition[0], partition[1], KNNClassifier(11))
    print 'Average accuracy: %f' % (acc_sum / 5.0)

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('\nReceived SIGINT...aborting.')
    except Exception:
        print >>sys.stderr, 'Received an unexpected exception:'
        sys.exit(traceback.format_exc())