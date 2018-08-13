from collections import namedtuple

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from lvqtoolbox.common import reshape_prototypes, restore_prototypes, init_prototypes


# GLVQ - Metrics
def _squared_euclidean(v1, v2):
    return np.sum((v1 - v2)**2)


def _squared_euclidean_grad(v1, v2):
    return -2 * (v2 - v1)


def _euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))


def _euclidean_grad(v1, v2):
    difference = v2 - v1
    return (-1 * difference) / np.sqrt(np.sum(difference**2))


# GLVQ - Cost functions
# TODO: naming... hmm
def _distances_difference(prototypes, p_labels, data, d_labels, metric):
    # Prototypes are the x in for the to be optimized f(x, *args)
    prototypes = restore_prototypes(prototypes, p_labels.size, data.shape[1])

    distances = cdist(data, prototypes, metric)

    ii_same = np.transpose(np.array([d_labels == prototype_label for prototype_label in p_labels]))
    ii_diff = ~ii_same

    dist_temp = np.where(ii_same, distances, np.inf)
    dist_same = dist_temp.min(axis=1)
    i_dist_same = dist_temp.argmin(axis=1)

    dist_temp = np.where(ii_diff, distances, np.inf)
    dist_diff = dist_temp.min(axis=1)
    i_dist_diff = dist_temp.argmin(axis=1)

    return dist_same, dist_diff, i_dist_same, i_dist_diff


def _relative_distance_difference_cost(prototypes, p_labels, data, d_labels, metric, *args):
    dist_same, dist_diff, _, _ = _distances_difference(prototypes, p_labels, data, d_labels, metric)
    return np.sum((dist_same - dist_diff) / (dist_same + dist_diff))


# TODO: Jacobian function of the cost function... depending on how the minimizers work.
# TODO: Hessian function of the cost function ... depending on how the mininimizers work.

def _relative_distance_difference_grad(prototypes, p_labels, data, d_labels, metric, metric_grad):
    dist_same, dist_diff, i_dist_same, i_dist_diff = _distances_difference(prototypes, p_labels, data, d_labels, metric)

    num_features = data.shape[1]
    num_prototypes = p_labels.size

    relative_dist = 2 / (dist_same + dist_diff)**2
    prototypes = restore_prototypes(prototypes, num_prototypes, num_features)
    gradient = np.zeros(prototypes.shape)

    # TODO: REMOVE
    step_size = 0.01

    for i_prototype in range(0, num_prototypes):
        ii_same = i_prototype == i_dist_same
        ii_diff = i_prototype == i_dist_diff

        relative_dist_diff = dist_diff[ii_same] * relative_dist[ii_same]
        relative_dist_same = (-1 * dist_same[ii_diff]) * relative_dist[ii_diff]

        grad_same = metric_grad(data[ii_same, :], prototypes[i_prototype, :])
        grad_diff = metric_grad(data[ii_diff, :], prototypes[i_prototype, :])

        relative_dist_diff = np.reshape(np.repeat(relative_dist_diff, grad_same.shape[1]), grad_same.shape)
        relative_dist_same = np.reshape(np.repeat(relative_dist_same, grad_diff.shape[1]), grad_diff.shape)

        gradient[i_prototype, :] = step_size * (relative_dist_diff * grad_same + relative_dist_same * grad_diff)

    return reshape_prototypes(gradient, num_prototypes, num_features)


class GLVQClassifier(BaseEstimator, ClassifierMixin):
    """GLVQ"""

    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, data, d_labels):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]
            The training input samples.
        d_labels : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        data, d_labels = check_X_y(data, d_labels)

        # Store shape of data for easy access

        # Check if p_labels are set in constructor
        # TODO: prototype labels should not be set in constructor... but the configuration should be dict/list ['class': num_prototypes]/[num_prototypes, etc] correspoinding with unique_labels
        self.p_labels_ = unique_labels(d_labels)
        self.prototypes_ = init_prototypes(self.p_labels_, data, d_labels)

        num_features = data.shape[1]
        num_prototypes = self.prototypes_.shape[0]

        optimize_results = minimize(_relative_distance_difference_cost,
                                    reshape_prototypes(self.prototypes_, num_prototypes, num_features),
                                    (self.p_labels_, data, d_labels, _squared_euclidean, _squared_euclidean_grad),
                                    'L-BFGS-B',
                                    options={'disp': True})

        self.prototypes_ = restore_prototypes(optimize_results.x, num_prototypes, num_features)
        # Return the classifier
        return self

    def predict(self, data):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        data : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of int of shape = [n_samples]
            The label for each sample is the label of the closest sample
            seen udring fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['prototypes_'])

        # Input validation
        data = check_array(data)

        # TODO: Map to prototypeLabels would be useful in the case the labels are not 0, 1, 2
        return cdist(data, self.prototypes_, _squared_euclidean).argmin(axis=1)

        # closest = np.argmin(euclidean_distances(data, self.prototypes_), axis=1)
        # return self.y_[closest]