from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels

import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from lvqtoolbox.common import init_prototypes
from lvqtoolbox.metrics import squared_euclidean, squared_euclidean_grad
from lvqtoolbox.scaling import sigmoid, sigmoid_grad
from lvqtoolbox.objective import relative_distance_difference_cost, relative_distance_difference_grad


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

        # Set these options in constructor and check if it's a string -> for existing functions or callable for custom
        # _sigmoid or _identity
        scalefun = sigmoid
        scalefun_grad = sigmoid_grad
        scalefun_kwargs = {'beta': 2}

        # _squared_euclidean or _euclidean
        metricfun = squared_euclidean
        metricfun_grad = squared_euclidean_grad

        # _relative_distance_difference_cost - GLVQ standard
        costfun = relative_distance_difference_cost
        costfun_args = (self.p_labels_,
                        data,
                        d_labels,
                        scalefun,
                        metricfun,
                        scalefun_grad,
                        metricfun_grad)
        costfun_grad = relative_distance_difference_grad

        self.optimize_results_ = minimize(costfun,
                                          self.prototypes_.ravel(),
                                          costfun_args,
                                          'L-BFGS-B')
                                          # costfun_grad,
                                          # options={'disp': False})

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
        return cdist(data, self.prototypes_, squared_euclidean).argmin(axis=1)

        # closest = np.argmin(euclidean_distances(data, self.prototypes_), axis=1)
        # return self.y_[closest]