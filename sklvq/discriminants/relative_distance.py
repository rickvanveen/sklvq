from . import DiscriminativeBaseClass

import numpy as np


class RelativeDistance(DiscriminativeBaseClass):

    # TODO: Look into why runtime warming occurs and what to do about it. Maybe the initialization of the
    #  prototypes is too close together?
    def __call__(self, dist_same, dist_diff):
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return (dist_same - dist_diff) / (dist_same + dist_diff)

    def gradient(self, dist_same, dist_diff):
        """ The partial derivative of the objective itself with respect to the prototypes (GLVQ) """
        with np.errstate(divide='ignore', invalid='ignore'):  # Suppresses runtime warning
            return 2 * dist_diff / (dist_same + dist_diff) ** 2
