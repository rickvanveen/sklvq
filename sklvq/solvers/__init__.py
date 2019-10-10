from ..misc import utils
from .solver_base_class import SolverBaseClass
from .scipy_base_solver import ScipyBaseSolver

ALIASES = {'bgd': 'batch-gradient-descent', 'lbfgs': 'lbfgs-solver', 'bfgs': 'bfgs-solver'}
BASE_CLASS = SolverBaseClass
PACKAGE = 'sklvq.solvers'


def grab(class_type, class_params):
    if class_type in ALIASES.keys():
        class_type = ALIASES[class_type]

    module_name, class_name = utils.process(class_type)

    return utils.find(PACKAGE, module_name, class_name, class_params, BASE_CLASS)
