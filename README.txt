Lab Assignment 2
Daniel Marchese <marchese.29@osu.edu>
Peter Jacobs <jacobs.269@osu.edu>
CSE 5243: Introduction to Data Mining - Dr. Srinivasan Parthasarathy


SUBMISSION CONTENTS
The submission contains all of the files from the project.  It contains at its root the driver
program for both of the classifiers, the data used for classification, a script for performing
cross-validation, and the code for the classifiers themselves


PROJECT STRUCTURE
The project is structured as follows
/
├── README.txt
├── classifiers
│   ├── __init__.py
│   ├── bayes.py
│   └── nearestneighbors.py
├── data
│   ├── 1000WordList.pkl
│   ├── 256WordList.pickle
│   ├── knnresults.txt
│   ├── reducedFeatureVectorMatrix1000.pkl
│   ├── reducedFeatureVectorMatrix256.pkl
│   ├── words.pickle
│   └── words.txt
├── knndriver.py
├── nbdriver.py
└── util
    ├── __init__.py
    └── crossValidation.py

Important files are outline below:

knndriver.py and nbdriver.py
These are the files that drive the logic for classification using the two implemented methods.

util/crossValidation.py
This script will partition a dataset to perfor five-fold cross validation.

data/
The data folder consists of several of the data files.  Among these files are the raw results from
the knn classifier's trials (knnresults.txt).  The meaning of the other data files becomes evident
after investigating the code.

classifiers/bayes.py
This is the location where all of the logic for running the Naive Bayes Classifier exists.  It was
implemented solely by Peter Jacobs.

classifiers/nearestneighbors.py
This is the location where all of the logic for running the K Nearest Neighbors Classifier exists.
It was implemented solely by Daniel Marchese.


RUNNING THE CODE
The code's only external dependency is numpy, so it is assumed that numpy exists on your python path
before execution.

RUNNING NAIVE BAYES
The Naive Bayes Classifier is configured to run all its tests using the driver.  From the root of
the project, run the following command (where > is the unix prompt).
> python nbdriver.py

RUNNING K NEAREST NEIGHBORS
The KNN classifier script only runs one classification at a time.  The provided configuration runs
5-fold cross validation on the set of non-null topic vectors using the paired-down feature vectors.
To run on the full feature vectors, you will need to comment out lines 45 and 46, and remove the
second argument to the function on line 49.  Running is as simple as:
> python knndriver.py


OWNERSHIP
crossValidation.py, nbdriver.py, and bayes.py were created and implemented by Peter Jacobs
knndriver.py, nearestneighbors.py, and the project structure were created by Daniel Marchese


CREDITS
The following open-source libraries are leveraged by this project.
 - numpy - http://www.numpy.org/
    - Used for fast euclidean distance between feature vectors in kNN classifier.
