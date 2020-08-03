import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV,
    RepeatedStratifiedKFold,
)
from sklearn.pipeline import make_pipeline

from sklvq import GLVQ


class ProgressLogger:
    def __init__(self):
        self.states = np.array([])

    def __call__(self, model, state) -> bool:
        self.states = np.append(self.states, state)
        return False


def test_glvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    labels = np.asarray(iris.target, str)

    progress_logger = ProgressLogger()

    classifier = GLVQ(
        solver_type="waypoint-gradient-descent",
        solver_params={"callback": progress_logger},
        distance_type="squared-euclidean",
        activation_type="swish",
        force_all_finite=True,
        random_state=31415,
    )
    classifier = classifier.fit(iris.data, labels)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == labels) / labels.size

    # print(progress_logger.costs)
    print("\nIris accuracy: {}".format(accuracy))

    

#
# def test_glvq_with_multiple_prototypes_per_class():
#     iris = datasets.load_iris()
#
#     iris.data = preprocessing.scale(iris.data)
#
#     classifier = GLVQ(
#         activation_type="sigmoid", activation_params={"beta": 3}, prototypes_per_class=4
#     )
#     classifier = classifier.fit(iris.data, iris.target)
#
#     predicted = classifier.predict(iris.data)
#
#     accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size
#
#     print("\nIris accuracy: {}".format(accuracy))
#
#
# # TODO: Have to look into scoring of CV and if they now provide the testing or training scores....
# def test_glvq_pipeline_iris():
#     iris = datasets.load_iris()
#
#     pipeline = make_pipeline(
#         preprocessing.StandardScaler(),
#         GLVQ(activation_type="sigmoid", activation_params={"beta": 6}),
#     )
#     accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
#     print("\nCross validation (k=5): " + "{}".format(accuracy))
#
#
# def test_glvq_gridsearch_iris():
#     iris = datasets.load_iris()
#
#     estimator = GLVQ()
#     pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)
#
#     param_grid = [
#         {
#             "glvq__solver_type": ["adam"],
#             "glvq__distance_type": ["squared-euclidean"],
#             "glvq__activation_type": ["sigmoid", "swish"],
#             "glvq__activation_params": [
#                 {"beta": beta} for beta in list(range(2, 10, 2))
#             ],
#         }
#     ]
#
#     search = GridSearchCV(
#         pipeline,
#         param_grid,
#         scoring="accuracy",
#         cv=5,
#         n_jobs=2,
#         return_train_score=True,
#     )
#
#     search.fit(iris.data, iris.target)
#
#     print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
#     print(search.best_params_)
#
#
# def test_glvq_gridsearch_all_iris():
#     iris = datasets.load_iris()
#
#     estimator = GLVQ()
#     pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)
#
#     solvers_types = [
#         "lbfgs",
#         "bfgs",
#         "steepest-gradient-descent",
#         "waypoint-gradient-descent",
#         "adaptive-moment-estimation",
#     ]
#
#     distance_types = ["squared-euclidean", "euclidean"]
#
#     activation_types = ["identity", "sigmoid", "soft-plus", "swish"]
#
#     param_grid = [
#         {
#             "glvq__solver_type": solvers_types,
#             "glvq__distance_type": distance_types,
#             "glvq__activation_type": activation_types,
#         }
#     ]
#
#     repeated_kfolds = RepeatedStratifiedKFold(n_splits=5, n_repeats=1)
#
#     search = GridSearchCV(
#         pipeline,
#         param_grid,
#         scoring="accuracy",
#         cv=repeated_kfolds,
#         n_jobs=4,
#         return_train_score=True,
#     )
#
#     search.fit(iris.data, iris.target)
#
#     print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
#     print(search.best_params_)
#
#
