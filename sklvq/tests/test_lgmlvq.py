import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import set_config


from sklvq import LGMLVQ


class ProgressLogger:
    def __init__(self):
        self.states = np.array([])

    def __call__(self, state: dict) -> bool:
        self.states = np.append(self.states, state)
        return False


def test_lgmlvq_iris():
    set_config(assume_finite=False)
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    progress_logger = ProgressLogger()

    classifier = LGMLVQ(
        solver_type="steepest-gradient-descent",
        solver_params={
            "callback": progress_logger,
            "max_runs": 20,
            "step_size": np.array([0.001, 0.5]),
            "batch_size": 25
        },
        activation_type="swish",
        localization="prototype",
        distance_type="local-adaptive-squared-euclidean",
        normalized_omega=False,
        force_all_finite=True
    )
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("Iris accuracy: {}".format(accuracy))
