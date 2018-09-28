import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline
import scipy as sp

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


def test_glvq_pipeline_iris():
    iris = datasets.load_iris()

    pipeline = make_pipeline(preprocessing.StandardScaler(), GLVQClassifier(scaling='sigmoid',
                                                                            beta=6))
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("Cross validation (k=5): " + "{}".format(accuracy))


def test_gmvlq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GMLVQClassifier(scaling='sigmoid', beta=2)
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_gmlvq_pipeline_iris():
    iris = datasets.load_iris()

    pipeline = make_pipeline(preprocessing.StandardScaler(), GMLVQClassifier(scaling='identity',
                                                                             beta=6))
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("Cross validation (k=5): " + "{}".format(accuracy))


def omega_gradient(data, prototype, omega):
    return np.apply_along_axis(lambda x, o: (o.dot(np.atleast_2d(x).T).dot(2 * np.atleast_2d(x))).ravel(),
                               1, (data - prototype), omega)


def test_omega_gradient():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototype = np.array([[1, 1, 1]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(omega_gradient(data, prototype, omega))


def gradient(data, prototype, omega):
    return np.apply_along_axis(lambda x, l: l.dot(np.atleast_2d(x).T).T,
                               1, (-2 * (data - prototype)), (omega.T.dot(omega))).squeeze()


def test_gradient():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototype = np.array([[1, 1, 1]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(gradient(data, prototype, omega).shape)


# Abusing the mahalanobis distance. Is probably much faster than antyhing that can be written in pure python
def distance(data, prototypes, omega):
    return sp.spatial.distance.cdist(data, prototypes, 'mahalanobis', VI=omega.T.dot(omega)) ** 2


def test_distance():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototypes = np.array([[1, 1, 1], [3, 3, 3]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))



    print(distance(data, prototypes, omega))