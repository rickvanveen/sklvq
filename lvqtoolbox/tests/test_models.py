import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline

from lvqtoolbox.models import GLVQClassifier
from lvqtoolbox.distance import sqeuclidean
from lvqtoolbox.scaling import identity, sigmoid


def test_glvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier(scalefun_options={'beta': 6},
                                prototypes_per_class=1,  # Broken
                                optimizer_options={'disp': True})
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_glvq_pipeline_iris():
    iris = datasets.load_iris()

    pipeline = make_pipeline(preprocessing.StandardScaler(), GLVQClassifier(distfun=sqeuclidean,
                                                                            scalefun=sigmoid,
                                                                            scalefun_options={'beta': 6}))
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("Cross validation (k=5): " + "{}".format(accuracy))


# TODO: Test if making everything callable does not screw up grid search
def test_glvq_gridsearch_iris():
    iris = datasets.load_iris()

    # grid = [{'glvqclassifier__scalingfun_param': ['identity']},
    #         {'glvqclassifier__scalingfun_param': ['sigmoid'],
    #          'glvqclassifier__scalingfun_options': [{'beta': x} for x in range(2,22,2)]}]
    # grid = ParameterGrid(grid)

    grid = [{'glvqclassifier__scalefun': [identity]},
            {'glvqclassifier__scalefun_options': [{'beta': 2}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 4}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 6}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 8}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 10}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 12}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 14}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 16}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 18}],
            'glvqclassifier__scalefun': [sigmoid]},
            {'glvqclassifier__scalefun_options': [{'beta': 20}],
            'glvqclassifier__scalefun': [sigmoid]}]

    pipeline = make_pipeline(preprocessing.StandardScaler(), GLVQClassifier())

    estimator = GridSearchCV(pipeline, grid, cv=10)
    result = estimator.fit(iris.data, iris.target)
    print("Done!")


def test_glvq_wine():
    wine = datasets.load_wine()

    wine.data = preprocessing.scale(wine.data)

    classifier = GLVQClassifier(optimizer_options={'disp': False})
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