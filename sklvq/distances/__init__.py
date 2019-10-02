from ..misc.utils import find

from .distance_base_class import DistanceBaseClass

# key equals the keyword and the value what the module/class are actually called
class_aliases = {'sqeuclidean': 'SquaredEuclidean'}
module_aliases = {'sqeuclidean': 'squared_euclidean'}


def create(distance_type, distance_params):
    return find(distance_type, distance_params, DistanceBaseClass, 'sklvq.distances',
                class_aliases=class_aliases,
                module_aliases=module_aliases)
