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

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_glvq_wine():
    wine = datasets.load_wine()

    wine.data = preprocessing.scale(wine.data)

    classifier = GLVQClassifier()
    classifier = classifier.fit(wine.data, wine.target)

    predicted = classifier.predict(wine.data)

    accuracy = np.count_nonzero(predicted == wine.target) / wine.target.size

    print("Wine accuracy: {}".format(accuracy))


def test_glvq_cancer():
    cancer = datasets.load_breast_cancer()

    cancer.data = preprocessing.scale(cancer.data)

    classifier = GLVQClassifier()
    classifier = classifier.fit(cancer.data, cancer.target)

    predicted = classifier.predict(cancer.data)

    accuracy = np.count_nonzero(predicted == cancer.target) / cancer.target.size

    print("Cancer accuracy: {}".format(accuracy))


def test_glvq_digits():
    digits = datasets.load_digits()

    digits.data = preprocessing.scale(digits.data)

    classifier = GLVQClassifier()
    classifier = classifier.fit(digits.data, digits.target)

    predicted = classifier.predict(digits.data)

    accuracy = np.count_nonzero(predicted == digits.target) / digits.target.size

    print("Digits accuracy: {}".format(accuracy))