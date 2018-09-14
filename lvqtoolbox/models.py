import numpy as np

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

from .common import init_prototypes
from .distance import sqeuclidean, sqeuclidean_grad, compute_distance
from .scaling import sigmoid, sigmoid_grad
from .objective import relative_distance_difference_cost


class GLVQClassifier(BaseEstimator, ClassifierMixin):
    """GLVQClassifier"""

    def __init__(self, scalefun=sigmoid, scalefun_grad=sigmoid_grad, scalefun_kwargs=None,
                 distfun=sqeuclidean, distfun_grad=sqeuclidean_grad, distfun_kwargs=None,
                 prototypes_per_class=1, optimizer='L-BFGS-B', optimizer_options=None, random_state=None):


        self.scalefun = scalefun # Cost function specific
        self.scalefun_grad = scalefun_grad # Cost function specific
        self.scalefun_kwargs = scalefun_kwargs # Cost function specific


        self.distfun = distfun # LVQ specific - all of them will have it
        self.distfun_grad = distfun_grad # LVQ specific
        self.distfun_kwargs = distfun_kwargs # LVQ specific
        self.prototypes_per_class = prototypes_per_class # LVQ specific
        self.optimizer = optimizer # LVQ specific
        self.optimizer_options = optimizer_options # LVQ specific
        self.random_state = random_state # sklearn specific

    def fit(self, data, y):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # SciKit-learn required check
        data, d_labels = check_X_y(data, y)

        # SciKit-learn required check
        check_classification_targets(d_labels)

        # SciKit-learn required check
        rng = check_random_state(self.random_state)

        # If no additional kwargs are given
        if self.scalefun_kwargs is None:
            self.scalefun_kwargs = {}

        # If no additional kwargs are given
        if self.distfun_kwargs is None:
            self.distfun_kwargs = {}

        # If no additional options dict (not kwargs for optimizer) is given
        if self.optimizer_options is None:
            self.optimizer_options = {}

        # TODO: Set classes_ to None in init? Compatibility problem with sklearn?
        self.classes_, d_labels = np.unique(d_labels, return_inverse=True) # TODO: How does this work?

        # TODO: Expect valid input... only check for cases where valid  input leads to incorrect output.
        if np.isscalar(self.prototypes_per_class):
            self.p_labels_ = np.repeat(unique_labels(d_labels), self.prototypes_per_class)
        # Assuming valid input: it is now a vector, which does not require any processing

        self.prototypes_ = init_prototypes(self.p_labels_, data, d_labels, rng)

        # This object implements the GLVQ standard _relative_distance_difference_cost objective/cost function
        costfun = relative_distance_difference_cost
        costfun_args = (self.p_labels_,
                        data,
                        d_labels,
                        self.scalefun,
                        self.distfun,
                        self.scalefun_grad,
                        self.distfun_grad,
                        self.scalefun_kwargs,
                        self.distfun_kwargs)
        costfun_grad = True # Bool tells minimize that the costfun return the cost and derivative can be callable

        self.optimize_results_ = minimize(costfun,
                                          self.prototypes_.ravel(),
                                          costfun_args,
                                          self.optimizer,
                                          costfun_grad,
                                          options=self.optimizer_options)

        # Assumes valid input for data and prototypes of shape = [n_observations/n_prototypes, n_features]
        num_features = data.shape[1]
        num_prototypes = self.prototypes_.shape[0]

        self.prototypes_ = self.optimize_results_.x.reshape([num_prototypes, num_features])
        # Return the classifier
        return self

    # # TODO: Score function (based on objective function?) Call this in predict? Necessary to get scores for ROC computation?
    def decision_function(self, data, d_labels):
        #decision function of shape (n_samples, n_classes) as all other classifiers
        check_is_fitted(self, ['prototypes_', 'p_labels_', 'distfun', 'distfun_kwargs'])

        dist_same, dist_diff, _, _ = compute_distance(self.prototypes_, self.p_labels_,
                                                      data, d_labels,
                                                      self.distfun, self.distfun_kwargs)
        return dist_diff.min(axis=1) - dist_same.min(axis=1) # TODO: this works differently...


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
        check_is_fitted(self, ['prototypes_', 'p_labels_', 'classes_'])

        # Input validation
        data = check_array(data)

        # TODO: Does not take the correct label of the prototype...
        return self.classes_.take(self.distfun(data, self.prototypes_).argmin(axis=1))

        # closest = np.argmin(euclidean_distances(data, self.prototypes_), axis=1)
        # return self.y_[closest]