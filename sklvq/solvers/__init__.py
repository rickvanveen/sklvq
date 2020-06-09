from .base import SolverBaseClass, ScipyBaseSolver
from .steepest_gradient_descent import SteepestGradientDescent
from .waypoint_gradient_descent import WaypointGradientDescent
from .limited_memory_bfgs import LimitedMemoryBfgs
from .broyden_fletcher_goldfarb_shanno import BroydenFletcherGoldfarbShanno
from .adaptive_moment_estimation import AdaptiveMomentEstimation
from ..misc import utils


__all__ = [
    "SolverBaseClass",
    "ScipyBaseSolver",
    "SteepestGradientDescent",
    "WaypointGradientDescent",
    "AdaptiveMomentEstimation",
    "BroydenFletcherGoldfarbShanno",
    "LimitedMemoryBfgs",
]

ALIASES = {
    "sgd": "steepest-gradient-descent",
    "lbfgs": "limited-memory-BFGS",
    "bfgs": "broyden-fletcher-goldfarb-shanno",
    "adadelta": "adaptive-gradient-descent",
    "adam": "adaptive-moment-estimation",
}
BASE_CLASS = SolverBaseClass
PACKAGE = "sklvq.solvers"


def grab(class_type, class_args=[], class_kwargs={}, whitelist=[]):
    return utils.grab(
        class_type, class_args, class_kwargs, ALIASES, whitelist, PACKAGE, BASE_CLASS
    )
