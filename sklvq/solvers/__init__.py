from ..misc import utils
from .solver_base_class import SolverBaseClass

ALIASES = {'bgd': 'batch-gradient-descent'}
BASE_CLASS = SolverBaseClass
PACKAGE = 'sklvq.solvers'


def grab(class_type, class_params):
    if class_type in ALIASES.keys():
        class_type = ALIASES[class_type]

    module_name, class_name = utils.process(class_type)

    return utils.find(PACKAGE, module_name, class_name, class_params, BASE_CLASS)
