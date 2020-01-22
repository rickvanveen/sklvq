import numpy as np

from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid
from sklearn.pipeline import make_pipeline

from sklvq import GLVQClassifier


def test_glvq_iris():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier(solver_type='steepest-gradient-descent', distance_type='squared-euclidean',
                                activation_type='identity', activation_params={'beta': 2})
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("\nIris accuracy: {}".format(accuracy))


# def test_glvq_custom_functions():
#
#     class Identity(ActivationBaseClass):
#         def __call__(self, x: np.ndarray) -> np.ndarray:
#             return x
#
#         def gradient(self, x: np.ndarray) -> np.ndarray:
#             return np.ones(x.shape)
#
#     test_activation_compatibility(Identity)
#
#     iris = datasets.load_iris()
#
#     iris.data = preprocessing.scale(iris.data)
#
#     classifier = GLVQClassifier(solver_type='steepest-gradient-descent', distance_type='sqeuclidean',
#                                 activation_type=Identity())
#     classifier = classifier.fit(iris.data, iris.target)
#
#     predicted = classifier.predict(iris.data)
#
#     accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size
#
#     print("\nIris accuracy: {}".format(accuracy))


def test_glvq_with_multiple_prototypes_per_class():
    iris = datasets.load_iris()

    iris.data = preprocessing.scale(iris.data)

    classifier = GLVQClassifier(activation_type='sigmoid', activation_params={'beta': 3}, prototypes_per_class=4)
    classifier = classifier.fit(iris.data, iris.target)

    predicted = classifier.predict(iris.data)

    accuracy = np.count_nonzero(predicted == iris.target) / iris.target.size

    print("\nIris accuracy: {}".format(accuracy))


# TODO: Have to look into scoring of CV and if they now provide the testing or training scores....
def test_glvq_pipeline_iris():
    iris = datasets.load_iris()

    pipeline = make_pipeline(preprocessing.StandardScaler(), GLVQClassifier(activation_type='sigmoid',
                                                                            activation_params={'beta': 6}))
    accuracy = cross_val_score(pipeline, iris.data, iris.target, cv=5)
    print("\nCross validation (k=5): " + "{}".format(accuracy))


def test_glvq_gridsearch_iris():
    iris = datasets.load_iris()

    estimator = GLVQClassifier()
    pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)

    # param_grid = [{'glvqclassifier__activation_type': ['identity'],
    #                'glvqclassifier__prototypes_per_class': [1, 2, 3]},
    #               {'glvqclassifier__activation_type': ['swish'],
    #                'glvqclassifier__activation_params': [{'beta': beta} for beta in list(range(2, 10, 2))],
    #                'glvqclassifier__prototypes_per_class': [1, 2, 3]},
    #               {'glvqclassifier__activation_type': ['soft+'],
    #                'glvqclassifier__activation_params': [{'beta': beta} for beta in list(range(2, 10, 2))],
    #                'glvqclassifier__prototypes_per_class': [1, 2, 3]}]

    param_grid = [{'glvqclassifier__solver_type': ['steepest-gradient-descent'],
                   'glvqclassifier__distance_type': ['squared-euclidean'],
                   # 'glvqclassifier__distance_params': [{'beta': beta} for beta in list(range(2, 10, 2))],
                   'glvqclassifier__activation_type': ['sigmoid', 'swish'],
                   'glvqclassifier__activation_params': [{'beta': beta} for beta in list(range(2, 10, 2))]}]

    search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=2, return_train_score=True)

    search.fit(iris.data, iris.target)

    print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)

# # TODO: TypeError: gradient() missing 1 required positional argument: 'x'... No idea where this happens.
# def test_glvq_gridsearch_iris_custom_functions():
#     class Identity(ActivationBaseClass):
#         def __call__(self, x: np.ndarray) -> np.ndarray:
#             return x
#
#         def gradient(self, x: np.ndarray) -> np.ndarray:
#             return np.ones(x.shape)
#
#     test_activation_compatibility(Identity)
#
#     iris = datasets.load_iris()
#     estimator = GLVQClassifier()
#     pipeline = make_pipeline(preprocessing.StandardScaler(), estimator)
#
#     param_grid = [{'glvqclassifier__solver_type': ['steepest-gradient-descent'],
#                    'glvqclassifier__distance_type': ['squared-euclidean'],
#                    'glvqclassifier__activation_type': ['sigmoid', 'swish', Identity],
#                    'glvqclassifier__activation_params': [{'beta': beta} for beta in list(range(2, 10, 2))]}]
#
#     search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5, n_jobs=2, return_train_score=True)
#
#     search.fit(iris.data, iris.target)
#
#     df = pd.DataFrame(search.cv_results_)
#     # df.to_clipboard()
#
#     print("\nBest parameter (CV score=%0.3f):" % search.best_score_)
#     print(search.best_params_)

# TODO: repeated gridsearch
# def test_glvq_repeated_gridsearch_iris():
#     pass
