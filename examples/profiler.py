import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn import set_config

from sklvq import GLVQClassifier
from sklvq import GMLVQClassifier
from sklvq import LGMLVQClassifier


set_config(assume_finite=True) # ~11% faster when true.
digits = datasets.load_digits()

digits.data = preprocessing.scale(digits.data)

# higher batch size is better for speed. Waypoint is much faster... due to batch size.
classifier = LGMLVQClassifier(
    solver_type="steepest-gradient-descent",
    solver_params={"max_runs": 20 , "step_size": np.array([0.2, 0.01]), "batch_size": 1},
    activation_type="identity",
)
classifier = classifier.fit(digits.data, digits.target)

predicted = classifier.predict(digits.data)

accuracy = np.count_nonzero(predicted == digits.target) / digits.target.size

print("Digits accuracy: {}".format(accuracy))
