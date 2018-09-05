import numpy as np

from scipy.optimize import minimize

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

from .common import init_prototypes
from .distance import sqeuclidean, sqeuclidean_grad
from .scaling import sigmoid, sigmoid_grad
from .objective import relative_distance_difference_cost


# TODO: MLPClassifier of sklearn also has option for solvers and just accepts all parameters and ignores them when a
# TODO: solver is used that doesn't use that option. Also options are strings
# TODO: Problem is that we do nto have only one function but multiple
class GLVQClassifier(BaseEstimator, ClassifierMixin):
    """GLVQClassifier"""

    def __init__(self, scalefun=sigmoid, scalefun_grad=sigmoid_grad, scalefun_kwargs=None,
                 distfun=sqeuclidean, distfun_grad=sqeuclidean_grad, distfun_kwargs=None,
                 prototypes_per_class=1, optimizer='L-BFGS-B', optimizer_options=None, random_state=None):
        self.scalefun = scalefun # Costfunction specific?
        self.scalefun_grad = scalefun_grad
        self.scalefun_kwargs = scalefun_kwargs
        self.distfun = distfun # LVQ in general - all of them will have it
        self.distfun_grad = distfun_grad
        self.distfun_options = distfun_kwargs
        self.prototypes_per_class = prototypes_per_class
        self.optimizer = optimizer
        self.optimizer_options = optimizer_options
        self.random_state = random_state

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

        # TODO: Set classes_ to None in init? Compatibility problem with sklearn?
        self.classes_, d_labels = np.unique(d_labels, return_inverse=True) # TODO: How does this work?

        # TODO: Expect valid input... only check for cases where valid  input leads to incorrect output.
        if np.isscalar(self.prototypes_per_class):
            self.p_labels_ = np.repeat(unique_labels(d_labels), self.prototypes_per_class)
        # Assuming valid input: it is now a vector, which does not require any processing

        self.prototypes_ = init_prototypes(self.p_labels_, data, d_labels, rng)

        # Assumes valid input for data and prototypes of shape = [n_observations/n_prototypes, n_features]
        num_features = data.shape[1]
        num_prototypes = self.prototypes_.shape[0]

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
                        self.distfun_options)
        costfun_grad = True # Bool tells minimize that the costfun return the cost and derivative can be callable

        self.optimize_results_ = minimize(costfun,
                                          self.prototypes_.ravel(),
                                          costfun_args,
                                          self.optimizer,
                                          costfun_grad,
                                          options=self.optimizer_options)

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

        return self.classes_[self.distfun(data, self.prototypes_).argmin(axis=1)]

        # closest = np.argmin(euclidean_distances(data, self.prototypes_), axis=1)
        # return self.y_[closest]