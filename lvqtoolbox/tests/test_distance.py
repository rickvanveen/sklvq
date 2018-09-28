import numpy as np
import scipy as sp


def omega_gradient(data, prototype, omega):
    return np.apply_along_axis(lambda x, o: (o.dot(np.atleast_2d(x).T).dot(2 * np.atleast_2d(x))).ravel(),
                               1, (data - prototype), omega)


def test_omega_gradient():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototype = np.array([[1, 1, 1]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(omega_gradient(data, prototype, omega))


def gradient(data, prototype, omega):
    return np.apply_along_axis(lambda x, l: l.dot(np.atleast_2d(x).T).T,
                               1, (-2 * (data - prototype)), (omega.T.dot(omega))).squeeze()


def test_gradient():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototype = np.array([[1, 1, 1]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    print(gradient(data, prototype, omega).shape)


# Abusing the mahalanobis distance. Is probably much faster than antyhing that can be written in pure python
def distance(data, prototypes, omega):
    return sp.spatial.distance.cdist(data, prototypes, 'mahalanobis', VI=omega.T.dot(omega)) ** 2


def test_distance():
    data = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    prototypes = np.array([[1, 1, 1], [3, 3, 3]])

    omega = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    omega = omega / np.sqrt(np.sum(np.diagonal(omega.T.dot(omega))))

    print(distance(data, prototypes, omega))