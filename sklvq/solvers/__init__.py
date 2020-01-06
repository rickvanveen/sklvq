from .base import (SolverBaseClass, ScipyBaseSolver)
from .steepest_gradient_descent import SteepestGradientDescent
from .adaptive_gradient_descent import AdaptiveGradientDescent
from .adaptive_moment_estimation import AdaptiveMomentEstimation

__all__ = ['SolverBaseClass',
           'ScipyBaseSolver',
           'SteepestGradientDescent',
           'AdaptiveGradientDescent',
           'AdaptiveMomentEstimation']

from ..misc import utils

ALIASES = {'sgd': 'steepest-gradient-descent',
           'lbfgs': 'lbfgs-solver',
           'bfgs': 'bfgs-solver',
           'adadelta': 'adaptive-gradient-descent',
           'adam': 'adaptive-moment-estimation'}
BASE_CLASS = SolverBaseClass
PACKAGE = 'sklvq.solvers'


def grab(class_type, class_params):
    return utils.grab(class_type, class_params, ALIASES, PACKAGE, BASE_CLASS)
