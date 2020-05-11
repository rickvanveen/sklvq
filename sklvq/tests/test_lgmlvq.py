import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import set_config


from sklvq import LGMLVQClassifier


def test_lgmlvq_iris():
    set_config(assume_finite=False)
    iris = datasets.load_digits()

    # iris.data[np.random.choice(150, 50, replace=False), 2] = np.nan

    data = preprocessing.scale(iris.data)

    classifier = LGMLVQClassifier(
        localization="p",
        prototypes_per_class=1,
        solver_type="steepest-gradient-descent",
        solver_params={"step_size": np.array([1.0, 0.05]), "max_runs": 5, "batch_size": 0},
        activation_type="identity",
        distance_type="local-adaptive-squared-euclidean",
    )
    classifier = classifier.fit(data, iris.target)

    predicted = classifier.predict(data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))
