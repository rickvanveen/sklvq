from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

from scipy.optimize import minimize

import numpy as np
import lvqtoolbox as sklvq

# TODO: bit pointles class or its more a solver class/function
class BaseLVQClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, objectivefun, objectivefunargs, objectivegradfun,
                 distfun, distfunargs, distgradfun, solver, solverargs):
        self._objectivefun = objectivefun
        self._objectivefunargs = objectivefunargs
        self._objectivegradfun = objectivegradfun
        self._distfun = distfun
        self._distfunargs = distfunargs
        self._distgradfun = distgradfun
        self._solver = solver
        self._solverargs = solverargs

    def _validate_required(self, data, y):
        # SciKit-learn required check
        data, y = check_X_y(data, y)

        # SciKit-learn required check
        check_classification_targets(y)

        self._rng = check_random_state(self.random_state)

        return data, y

    def _fit(self, data, y):

        _objectivefunargs = (data, y) + self._objectivefunargs

        self.solver_results_ = minimize(self._objectivefun,
                                        self._variables,
                                        _objectivefunargs,
                                        self._solver,
                                        self._objectivegradfun,
                                        options=self._solverargs)


class GLVQClassifier(BaseLVQClassifier):

    # Assuming fixed cost function
    def __init__(self, distfun='sqeuclidean', scalefun='sigmoid', beta=2,
                 solver='L-BFGS-B', solverargs={}, prototypes_per_class=1, random_state=None):
        self.distfun = distfun
        self.scalefun = scalefun
        self.beta = beta
        self.prototypes_per_class = prototypes_per_class
        self.random_state = random_state
        self.solver = solver
        self.solverargs = solverargs

    # TODO: Get this from some configuration kind of file... depending on costfunction?
    def _validate_parameters(self):
        # TODO: check if string or callable
        if self.distfun == 'sqeuclidean':
            self._distfun = sklvq.distance.sqeuclidean
            self._distgradfun = sklvq.distance.sqeuclidean_grad
            self._distfunargs = {}
        elif self.distfun == 'euclidean':
            self._distfun = sklvq.distance.euclidean
            self._distgradfun = sklvq.distance.sqeuclidean_grad
            self._distfunargs = {}
        else:
            print('Something wrong with distance function')

        # TODO: Check if string of callable
        if self.scalefun == 'sigmoid':
            self._scalefun = sklvq.scaling.sigmoid
            self._scalegradfun = sklvq.scaling.sigmoid_grad
            self._scalfunargs = {'beta': self.beta}
        elif self.scalefun == 'identity':
            self._scalefun = sklvq.scaling.identity
            self._scalegradfun = sklvq.scaling.identity_grad
            self._scalfunargs = {}
        else:
            print('Something wrong with scaling function')

    def fit(self, data, y):
        # init self.variables_
        self._validate_parameters()

        # super(GLVQClassifier, self).__init__(r, objectivefunargs, objectivegradfun,
        #          distfun, distfunargs, distgradfun, solver, solverargs)

        data, y = self._validate_required(data, y)

        if np.isscalar(self.prototypes_per_class):
            self.prototype_y_ = np.repeat(unique_labels(y), self.prototypes_per_class)

        self._variables = sklvq.common.init_prototypes(self.prototype_y_, data, y, self._rng).ravel()

        solution = self._fit(data, y)
