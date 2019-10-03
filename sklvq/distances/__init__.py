from ..misc import utils

from .distance_base_class import DistanceBaseClass

# key equals the keyword and the value what the module/class are actually called
ALIASES = {'sqeuclidean': 'squared-euclidean'}
BASE_CLASS = DistanceBaseClass
PACKAGE = 'sklvq.distances'


def grab(class_type, class_params):
    if class_type in ALIASES.keys():
        class_type = ALIASES[class_type]

    module_name, class_name = utils.process(class_type)

    return utils.find(PACKAGE, module_name, class_name, class_params, BASE_CLASS)
