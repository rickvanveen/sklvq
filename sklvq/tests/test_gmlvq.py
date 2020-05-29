import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import set_config


from sklvq import GMLVQ


def test_gmlvq_iris():
    set_config(assume_finite=False)
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GMLVQ(
        solver_type="waypoint-gradient-descent",
        solver_params={"step_size": np.array([0.1, 0.01]), "max_runs": 20},
        activation_type="swish",
        activation_params={"beta": 2},
        distance_type="adaptive-squared-euclidean",
    )
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_gmlvq_with_multiple_prototypes_per_class():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GMLVQ(
        activation_type="sigmoid", activation_params={"beta": 6}, prototypes_per_class=4
    )
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)
    classifier.transform(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))


def test_gmlvq_pipeline_iris():
    iris = datasets.load_iris()

    pipeline = make_pipeline(
        preprocessing.StandardScaler(),
        GMLVQ(activation_type="sigmoid", activation_params={"beta": 6}),
    )
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("Cross validation (k=5): " + "{}".format(accuracy))


def test_gmlvq_gridsearch_iris():
    iris = datasets.load_iris()

    estimator = GMLVQ()
    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    param_grid = [
        {
            "gmlvq__solver_type": ["steepest-gradient-descent", "lbfgs"],
            "gmlvq__activation_type": ["sigmoid"],
            "gmlvq__activation_params": [{"beta": 2}],
        }
    ]

    search = GridSearchCV(pipeline, param_grid, scoring="accuracy", cv=5, n_jobs=2)

    search.fit(iris.data, iris.target)

    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
