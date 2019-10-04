from . import DistanceBaseClass

import numpy as np
import scipy as sp


class AdaptiveSquaredEuclidean(DistanceBaseClass):

    # TODO: make the model do this when updating omega
    def normalise(self):
        self.omega = self.omega / np.sqrt(np.sum(np.diagonal(self.omega.T.dot(self.omega))))

    def __call__(self, data, model):
        """ Implements a weighted variant of the squared euclidean distance.

                Note uses scipy.spatial.distance.cdist see scipy documentation.

                Note that any custom function should still accept and return the same as this function.

                Parameters
                ----------
                data       : ndarray, shape = [n_obervations, n_features]
                             Inputs are converted to float type.
                prototypes : ndarray, shape = [n_prototypes, n_features]
                             Inputs are converted to float type.
                omega      : ndarray, shape = [n_features, n_features]
                TODO: Rectangular Omega... nfeatures, x or x, nfeatures

                Returns
                -------
                distances : ndarray, shape = [n_observations, n_prototypes]
                    The dist(u=XA[i], v=XB[j]) is computed and stored in the
                    ij-th entry.
            """
        return sp.spatial.distance.cdist(data, model.prototypes_, 'mahalanobis', VI=model.omega_.T.dot(model.omega_)) ** 2

    # Returns: shape = [num_samples, num_features]
    @staticmethod
    def _prototype_gradient(data, prototype, omega):
        return np.apply_along_axis(lambda x, l: l.dot(np.atleast_2d(x).T).T,
                                   1, (-2 * (data - prototype)), (omega.T.dot(omega))).squeeze()

    # Returns: shape = [num_samples, omega.size]
    @staticmethod
    def _omega_gradient(data, prototype, omega):
        return np.apply_along_axis(lambda x, o: (o.dot(np.atleast_2d(x).T).dot(2 * np.atleast_2d(x))).ravel(),
                                   1, (data - prototype), omega)

    # TODO: Gradient should just give the gradient with respect to prototypes and omega not with respect to the
    #  prototypes only and then a separate for omega...

    # TODO the i_prototype will give problems when considering local Relevance per class and not per prototype. Then
    #  omega is not necessarily coupled to the prototype. But if this is set in the model we can do some processing
    #  here.
    def gradient(self, data, model, i_prototype):
        pass

    # TODO: now fix the problem of how to give back the gradient... Here we now how construct a generic gradient from
    #  the prototype and omega gradient... do we actually need them at the same time? Yes I think.
