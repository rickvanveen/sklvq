from .base import SolverBaseClass, ScipyBaseSolver
from .steepest_gradient_descent import SteepestGradientDescent
from .waypoint_gradient_descent import WaypointGradientDescent
from .lbfgs_solver import LbfgsSolver
from .bfgs_solver import BfgsSolver
from .adaptive_moment_estimation import AdaptiveMomentEstimation
from ..misc import utils


__all__ = [
    "SolverBaseClass",
    "ScipyBaseSolver",
    "SteepestGradientDescent",
    "WaypointGradientDescent",
    "AdaptiveMomentEstimation",
    "LbfgsSolver",
    "BfgsSolver",
]

ALIASES = {
    "sgd": "steepest-gradient-descent",
    "lbfgs": "lbfgs-solver",
    "bfgs": "bfgs-solver",
    "adadelta": "adaptive-gradient-descent",
    "adam": "adaptive-moment-estimation",
}
BASE_CLASS = SolverBaseClass
PACKAGE = "sklvq.solvers"


def grab(class_type, class_params):
    return utils.grab(class_type, class_params, ALIASES, PACKAGE, BASE_CLASS)
