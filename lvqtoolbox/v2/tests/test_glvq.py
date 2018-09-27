import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline


from lvqtoolbox.v2.models import GLVQClassifier


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

    classifier = GLVQClassifier(scaling='sigmoid', beta=6, prototypes_per_class=6)
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