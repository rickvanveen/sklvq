import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import set_config


from sklvq import TGMLVQClassifier

set_config(assume_finite=True)
iris = datasets.load_digits()

iris.data = preprocessing.scale(iris.data)

classifier = GMLVQClassifier(solver_type='waypoint-gradient-descent',
                             solver_params={'max_runs': 12, 'step_size': np.array([0.2, 0.01])},
                             activation_type='identity')
classifier = classifier.fit(iris.data, iris.target)

predicted = classifier.predict(iris.data)

accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

print("Iris accuracy: {}".format(accuracy))