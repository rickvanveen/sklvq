import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import set_config

from sklvq import GLVQClassifier


set_config(assume_finite=True)
iris = datasets.load_digits()

iris.data = preprocessing.scale(iris.data)

classifier = GLVQClassifier(solver_type='steepest-gradient-descent',
                            solver_params={'max_runs': 100, 'step_size': np.array([0.2])},
                            activation_type='identity')
classifier = classifier.fit(iris.data, iris.target)

predicted = classifier.predict(iris.data)

accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

print("Iris accuracy: {}".format(accuracy))