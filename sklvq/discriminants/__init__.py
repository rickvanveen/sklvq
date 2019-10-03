from ..misc import utils

from .discriminant_base_class import DiscriminativeBaseClass

ALIASES = {'reldist': 'relative-distance'}
BASE_CLASS = DiscriminativeBaseClass
PACKAGE = 'sklvq.discriminants'


def grab(class_type, class_params):
    if class_type in ALIASES.keys():
        class_type = ALIASES[class_type]

    module_name, class_name = utils.process(class_type)

    return utils.find(PACKAGE, module_name, class_name, class_params, BASE_CLASS)