from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import numpy as np

from scipy.spatial.distance import cdist


# GLVQ - Metrics


def _squared_euclidean(v1, v2):
    return np.sum((v1 - v2) ** 2)


def _squared_euclidean_grad(v1, v2):
    return -2 * (v2 - v1)


def _euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def _euclidean_grad(v1, v2):
    difference = v2 - v1
    return (-1 * difference) / np.sqrt(np.sum(difference ** 2))


# GLVQ - Cost functions

# TODO:
# TODO: not todo but more consistent with sklearn and theory: W, W_y, w, w_y, X, X_y, x, x_y but not with pep8
def _relative_distance_difference_cost(prototypes, prototypes_labels, data, data_labels, metricfun):
    # Prototypes are the x in for the to be optimized f(x, *args)
    prototypes = prototypes.reshape([prototypes_labels.size, data.shape[1]])

    distances = cdist(prototypes, data, metricfun)

    ii_same = np.array([data_labels == prototype_label for prototype_label in prototypes_labels])
    ii_diff = ~ii_same

    distance_same = np.where(ii_same, distances, np.inf).min(axis=1)
    distance_diff = np.where(ii_diff, distances, np.inf).min(axis=1)

    return np.sum((distance_same - distance_diff) / (distance_same + distance_diff))


# TODO: Gradient function of the cost function... depending on how the minimizers work.
# TODO: Hessian function of the cost function ... depending on how the mininimizers work.
# def std_costfun_grad(modelObject, dataObject):
# TODO: PrototypeClass?


class GLVQClassifier(BaseEstimator, ClassifierMixin):
    """GLVQ"""
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self.w_ = np.mean(self.X_)
        self.w_y_ = self.classes_


        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]