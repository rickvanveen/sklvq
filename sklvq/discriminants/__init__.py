from ..misc.utils import find
from .discriminant_base_class import DiscriminativeBaseClass

class_aliases = {'reldist': 'RelativeDistance'}
module_aliases = {'reldist': 'relative_distance'}


def create(discriminative_type, discriminative_params):
    return find(discriminative_type, discriminative_params, DiscriminativeBaseClass, 'sklvq.discriminants',
                class_aliases=class_aliases, module_aliases=module_aliases)