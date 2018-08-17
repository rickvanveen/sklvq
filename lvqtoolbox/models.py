import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

from .common import init_prototypes
from .metrics import sqeuclidean_grad, euclidean_grad
from .scaling import sigmoid, sigmoid_grad, identity, identity_grad
from .objective import relative_distance_difference_cost


class GLVQClassifier(BaseEstimator, ClassifierMixin):
    """GLVQClassifier"""

    # TODO: Make costfunction a parameter, but the rest (except optimizer?) depends on this so should be a of_args dict?
    def __init__(self, scalingfun_param='sigmoid', scalingfun_options=None,
                 metricfun_param='sqeuclidean', metricfun_options=None, prototypes_per_class=1,
                 optimizer='L-BFGS-B', optimizer_options=None, random_state=None):
        self.scalingfun_param = scalingfun_param
        self.scalingfun_options = scalingfun_options
        self.metricfun_param = metricfun_param
        self.metricfun_options = metricfun_options
        self.prototypes_per_class = prototypes_per_class
        self.optimizer = optimizer
        self.optimizer_options=optimizer_options
        self.random_state=random_state

    # def __init(self, demo_param='demo'):
    #     self.demo_param = demo_param

    def fit(self, data, y):
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
        # Check that data and y have correct shape
        data, d_labels = check_X_y(data, y)

        # Have to do this for sklearn compatibility
        check_classification_targets(d_labels)

        self.classes_, d_labels = np.unique(d_labels, return_inverse=True)

        if np.isscalar(self.prototypes_per_class):
            self.p_labels_ = np.repeat(unique_labels(d_labels), self.prototypes_per_class)
        elif isinstance(self.prototypes_per_class, np.ndarray):
            pass
        else:
            raise ValueError("Expected 'prototypes_per_class' to be a scalar or vector with value(s) > 0")

        rng = check_random_state(self.random_state)
        self.prototypes_ = init_prototypes(self.p_labels_, data, d_labels, rng)

        num_features = data.shape[1]
        num_prototypes = self.prototypes_.shape[0]

        # Set these options in constructor and check if it's a string -> for existing functions or callable for custom
        # _sigmoid or _identity

        # self.scalingfun_param TODO: support for custom one
        expected_scalingfuns = {'identity': (identity, identity_grad),
                                'sigmoid': (sigmoid, sigmoid_grad)}
        scalefun, scalefun_grad = expected_scalingfuns.get(self.scalingfun_param, (None, None))
        if scalefun is None or scalefun_grad is None:
            raise ValueError("Expected 'scalingfun_param' to be of the following: \n \t" +
                             ", ".join(expected_scalingfuns))

        # TODO: validate scaling fun kwargs (inspect.getargspec())
        # if self.scalingfun_param == 'sigmoid':
        #     if not isinstance(self.scalingfun_options, dict):
        #         raise ValueError("Expected 'scalingfun_options' to be an instance of Python dictionary.")
        #     # for key, value in self.scalingfun_options.items():
        # else:
        #     pass
        scalingfun_kwargs = self.scalingfun_options
        if self.scalingfun_options is None:
            scalingfun_kwargs = {}

        expected_metricfuns = {'sqeuclidean': ('sqeuclidean', sqeuclidean_grad),
                               'euclidean': ('euclidean', euclidean_grad)}
        metricfun, metricfun_grad = expected_metricfuns.get(self.metricfun_param, (None, None))
        if metricfun is None or metricfun_grad is None:
            raise ValueError("Expected 'metricfun_param' to be on of the following: \n \t" +
                             ", ".join(expected_metricfuns))

        # TODO: validate metric fun kwargs (inspect.getargspec())
        metricfun_kwargs = self.metricfun_options
        if self.metricfun_options is None:
            metricfun_kwargs = {}

        expected_optimizers = {'L-BFGS-B': 'L-BFGS-B', 'CG': 'CG'}
        optimizer = expected_optimizers.get(self.optimizer, None)
        if optimizer is None:
            raise ValueError("Expected 'optimizer' to be one of the following: \n \t" + ", ".join(expected_optimizers))

        optimizer_kwargs = self.optimizer_options
        if self.optimizer_options is None:
            optimizer_kwargs = {}

        # _relative_distance_difference_cost - GLVQ standard
        costfun = relative_distance_difference_cost
        costfun_args = (self.p_labels_,
                        data,
                        d_labels,
                        scalefun,
                        metricfun,
                        scalefun_grad,
                        metricfun_grad,
                        scalingfun_kwargs,
                        metricfun_kwargs)
        costfun_grad = True

        self.optimize_results_ = minimize(costfun,
                                          self.prototypes_.ravel(),
                                          costfun_args,
                                          optimizer,
                                          costfun_grad,
                                          options=optimizer_kwargs)

        self.prototypes_ = self.optimize_results_.x.reshape([num_prototypes, num_features])
        # Return the classifier
        return self

    # # TODO: Score function (based on objective function?) Call this in predict?
    # def decision_function(self, data):
    #     pass

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
        return self.classes_[cdist(data, self.prototypes_, self.metricfun_param).argmin(axis=1)]

        # closest = np.argmin(euclidean_distances(data, self.prototypes_), axis=1)
        # return self.y_[closest]