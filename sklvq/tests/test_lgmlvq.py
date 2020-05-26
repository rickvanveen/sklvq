import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import set_config


from sklvq import LGMLVQ


def test_lgmlvq_iris():
    set_config(assume_finite=False)
    iris = datasets.load_digits()

    # iris.data[np.random.choice(150, 50, replace=False), 2] = np.nan

    data = preprocessing.scale(iris.data)

    classifier = LGMLVQ(
        localization="c",
        prototypes_per_class=1,
        solver_type="steepest-gradient-descent",
        solver_params={"step_size": np.array([0.01, 0.001]), "max_runs": 10, "batch_size": 0},
        activation_type="sigmoid",
        activation_params={"beta": 2},
        distance_type="local-adaptive-squared-euclidean",
    )
    classifier = classifier.fit(data, iris.target)

    predicted = classifier.predict(data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))
