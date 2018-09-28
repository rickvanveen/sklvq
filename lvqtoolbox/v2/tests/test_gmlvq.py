import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline


from lvqtoolbox.v2.models import GMLVQClassifier


def test_gmlvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GMLVQClassifier(scaling='sigmoid', beta=2)
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_gmlvq_with_multiple_prototypes_per_class():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GMLVQClassifier(scaling='sigmoid', beta=6, prototypes_per_class=4)
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


# TODO: Have to look into scoring of CV and if they now provide the testing or training scores....
def test_gmlvq_pipeline_iris():
    iris = datasets.load_iris()

    # TODO: Supposed to work with make_pipline because  of not in __all__ error
    pipeline = make_pipeline(preprocessing.StandardScaler(), GMLVQClassifier(scaling='sigmoid',
                                                                            beta=6))
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("Cross validation (k=5): " + "{}".format(accuracy))


# TODO: Gridsearch