from .base import (SolverBaseClass, ScipyBaseSolver)
from .steepest_gradient_descent import SteepestGradientDescent


__all__ = ['SolverBaseClass',
           'ScipyBaseSolver',
           'SteepestGradientDescent']

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
