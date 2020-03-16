from . import DistanceBaseClass

import numpy as np
import scipy as sp

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class AdaptiveEuclidean(DistanceBaseClass):

    def __call__(self, data: np.ndarray, model: 'LVQClassifier') -> np.ndarray:
        """ Implements a weighted variant of the euclidean distance:
            .. math::
                d^{\\Lambda}(w, x) = \\sqrt{(x - w)^T \\Lambda (x - w)}

        .. note::
            Uses scipy.spatial.distance.cdist, see scipy documentation for more detail.

        Parameters
        ----------
        data : ndarray
            A matrix containing the samples on the rows.
        model : LVQClassifier
            In principle, any LVQClassifier that calls it's matrix omega.
            Specifically here, GMLVQClassifier.

        Returns
        -------
        ndarray
            The adaptive squared euclidean distance for every sample to every prototype stored row-wise.
        """
        return sp.spatial.distance.cdist(data, model.prototypes_, 'mahalanobis',
                                         VI=model.omega_.T.dot(model.omega_))

    def gradient(self, data: np.ndarray, model: 'LVQClassifier', i_prototype: int) -> np.ndarray:
        pass