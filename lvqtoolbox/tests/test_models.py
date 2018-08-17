import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline

from lvqtoolbox.models import GLVQClassifier


def test_glvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier(metricfun_param='sqeuclidean',
                                scalingfun_param='sigmoid',
                                scalingfun_options={'beta': 6},
                                prototypes_per_class=1, # Broken
                                optimizer='CG',
                                optimizer_options={'disp': True})
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_glvq_pipeline_iris():
    iris = datasets.load_iris()

    pipeline = make_pipeline(preprocessing.StandardScaler(), GLVQClassifier(metricfun_param='sqeuclidean',
                                                                            scalingfun_param='sigmoid',
                                                                            scalingfun_options={'beta': 6}))
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("Cross validation (k=5): " + "{}".format(accuracy))


# def test_glvq_gridsearch_iris():
#     iris = datasets.load_iris()
#
#     # TODO: AAAH uses setparam and that does not change sigmoid_grad when they set sigmoid
#     grid = [{'glvqclassifier__scalingfun_param': ['identity']},
#             {'glvqclassifier__scalingfun_param': ['sigmoid'],
#              'glvqclassifier__scalingfun_options': list(range(2, 22, 2))}]
#     grid = ParameterGrid(grid)
#
#     pipeline = make_pipeline(preprocessing.StandardScaler(), GLVQClassifier())
#
#     estimator = GridSearchCV(pipeline, grid.param_grid, cv=5)
#     estimator.fit(iris.data, iris.target)

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