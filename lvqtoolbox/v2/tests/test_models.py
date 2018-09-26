import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline

from lvqtoolbox.v2.models import GLVQClassifier, GMLVQClassifier


def test_glvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier(scaling='sigmoid', beta=6)
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_glvq_with_multiple_prototypes_per_class():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier(scaling='sigmoid', beta=6, prototypes_per_class=5)
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_gmvlq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GMLVQClassifier(scaling='identity')
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def omega_gradient(data, prototype, omega):
    return np.apply_along_axis(lambda x, o: o.dot(np.atleast_2d(x).T).dot(2 * np.atleast_2d(x)).shape,
                               1, (data - prototype), omega)


def test_omega_gradient():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototype = np.array([[1, 1, 1]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(omega_gradient(data, prototype, omega))