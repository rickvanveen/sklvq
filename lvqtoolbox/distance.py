from abc import ABC, abstractmethod
import scipy as sp
import numpy as np


class AbstractDistance(ABC):

    @abstractmethod
    def __call__(self, data, prototypes):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data, prototype):
        raise NotImplementedError("You should implement this!")


class RelevanceSquaredEuclidean(AbstractDistance):

    # TODO: Why set omega in the init but not prototypes etc... Omega should also not be set in the init
    def __init__(self, omega=None):
        self.omega = omega

    # TODO: make omega a property and normalise everytime 'automatically' when it is set?
    def normalise(self):
        self.omega = self.omega / np.sqrt(np.sum(np.diagonal(self.omega.T.dot(self.omega))))

    def __call__(self, data, prototypes):
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
        return sp.spatial.distance.cdist(data, prototypes, 'mahalanobis', VI=self.omega.T.dot(self.omega)) ** 2

    # Returns: shape = [num_samples, num_features]
    def gradient(self, data, prototype):
        return np.apply_along_axis(lambda x, l: l.dot(np.atleast_2d(x).T).T,
                                   1, (-2 * (data - prototype)), (self.omega.T.dot(self.omega))).squeeze()

    # Returns: shape = [num_samples, omega.size]
    def omega_gradient(self, data, prototype):
        return np.apply_along_axis(lambda x, o: (o.dot(np.atleast_2d(x).T).dot(2 * np.atleast_2d(x))).ravel(),
                                   1, (data - prototype), self.omega)

    # TODO: Gradient should just give the gradient with respect to prototypes and omega not with respect to the
    #  prototypes only and then a separate for omega...
