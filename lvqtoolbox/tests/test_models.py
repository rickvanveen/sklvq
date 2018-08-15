import scipy as sp
import numpy as np
from sklearn import datasets
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing

from lvqtoolbox.glvq import GLVQClassifier


def test_glvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier()
    classifier = classifier.fit(iris.data, iris.target)

    # print(classifier.prototypes_)
    # print(classifier.optimize_results_.fun)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_glvq_wine():
    wine = datasets.load_wine()

    wine.data = preprocessing.scale(wine.data)

    classifier = GLVQClassifier()
    classifier = classifier.fit(wine.data, wine.target)

    # print(classifier.prototypes_)
    # print(classifier.optimize_results_.fun)

    predicted = classifier.predict(wine.data)

    accuracy = np.count_nonzero(predicted == wine.target) / wine.target.size

    print("Wine accuracy: {}".format(accuracy))

