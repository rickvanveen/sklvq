from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels

import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from lvqtoolbox.common import init_prototypes


# GLVQ - Metrics
def _squared_euclidean(v1, v2):
    return np.sum((v1 - v2)**2)


def _squared_euclidean_grad(v1, v2):
    return -2 * (v1 - v2)


def _euclidean(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))


def _euclidean_grad(v1, v2):
    difference = v1 - v2
    return (-1 * difference) / np.sqrt(np.sum(difference**2))


# GLVQ - Cost functions
# TODO: naming... hmm maybe divide this up into multiple functions
def _distances_difference(prototypes, p_labels, data, d_labels, metric):
    distances = cdist(data, prototypes, metric)

    ii_same = np.transpose(np.array([d_labels == prototype_label for prototype_label in p_labels]))
    ii_diff = ~ii_same

    # TODO: Function
    dist_temp = np.where(ii_same, distances, np.inf)
    dist_same = dist_temp.min(axis=1)
    i_dist_same = dist_temp.argmin(axis=1)

    dist_temp = np.where(ii_diff, distances, np.inf)
    dist_diff = dist_temp.min(axis=1)
    i_dist_diff = dist_temp.argmin(axis=1)

    return dist_same, dist_diff, i_dist_same, i_dist_diff


# GLVQ - mu(x) functions
def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


# derivative mu(x) in glvq paper, same and diff are relative to the currents prototype's label
def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff)**2


# GLVQ - f(x) monotonically increasing functions
def _sigmoid(relative_distance):
    return 1 / (1 + np.exp( -1 * relative_distance))


def _sigmoid_grad(relative_distance):
    sigmoid = _sigmoid(relative_distance)
    return sigmoid * (1 - sigmoid)


# f(x) = x
def _identity(relative_distance):
    return relative_distance


# derivative f(x) = x is one TODO: necessary to be configurable?
def _identity_grad(*args):
    return 1


# TODO: f, and metric should be configurable. mu: relative distance also?
def _relative_distance_difference_cost(prototypes, p_labels,
                                       data, d_labels,
                                       scalefun, metricfun, *args):
    # Prototypes are the x in for the to be optimized f(x, *args)
    prototypes = prototypes.reshape(p_labels.size, data.shape[1])
    dist_same, dist_diff, _, _ = _distances_difference(prototypes, p_labels, data, d_labels, metricfun)
    return np.sum(scalefun(_relative_distance(dist_same, dist_diff)))


# TODO: Maybe change some of the functions to objects/classes better organized and more clear about how to extend. E.g.
# Scale.value(), Scale.gradient(), Metric.value(), Metric.gradient() etc... but maybe not...
def _relative_distance_difference_grad(prototypes, p_labels,
                                       data, d_labels,
                                       scalefun, metricfun,
                                       scalefun_grad, metricfun_grad):
    num_features = data.shape[1]
    num_prototypes = p_labels.size

    prototypes = prototypes.reshape([num_prototypes, num_features])
    dist_same, dist_diff, i_dist_same, i_dist_diff = _distances_difference(prototypes,
                                                                           p_labels, data,
                                                                           d_labels, metricfun)
    gradient = np.zeros(prototypes.shape)

    # TODO: REMOVE
    step_size = 0.05

    for i_prototype in range(0, num_prototypes):
        ii_same = i_prototype == i_dist_same
        ii_diff = i_prototype == i_dist_diff

        # f'(mu(x)) * (2 * d_2(x) / (d_1(x) + d_2(x))^2)
        relative_dist_same = scalefun_grad(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) * \
                             _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same])
        relative_dist_diff = scalefun_grad(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff])) * \
                             _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff])
        # -2 * (x - w)
        grad_same = metricfun_grad(data[ii_same, :], prototypes[i_prototype, :])
        grad_diff = metricfun_grad(data[ii_diff, :], prototypes[i_prototype, :])

        gradient[i_prototype, :] = step_size * (relative_dist_same @ grad_same - relative_dist_diff @ grad_diff)

    return gradient.ravel()


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

        # _sigmoid or _identity
        scalefun = _sigmoid
        scalefun_grad = _sigmoid_grad

        # _squared_euclidean or _euclidean
        metricfun = _squared_euclidean
        metricfun_grad = _squared_euclidean_grad

        # _relative_distance_difference_cost - GLVQ standard
        costfun = _relative_distance_difference_cost
        costfun_args = (self.p_labels_,
                        data,
                        d_labels,
                        scalefun,
                        metricfun,
                        scalefun_grad,
                        metricfun_grad)
        costfun_grad = _relative_distance_difference_grad

        self.optimize_results_ = minimize(costfun,
                                          self.prototypes_.ravel(),
                                          costfun_args,
                                          'L-BFGS-B',
                                          costfun_grad,
                                          options={'disp': True})

        self.prototypes_ = self.optimize_results_.x.reshape([num_prototypes, num_features])
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