# TODO: LVQClassifier - base class for LVQClassifier
# Abstract classes are not Pythonian apparently. However, common functionality which we always need can still be put in
# a common base class.

import numpy as np
import scipy as sp


from scipy.spatial.distance import cdist


def _conditional_mean(p_labels, data, d_labels):
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


def init_prototypes(p_labels, data, d_labels, rng):
    conditional_mean = _conditional_mean(p_labels, data, d_labels)
    return conditional_mean + (1e-4 * rng.uniform(-1, 1, conditional_mean.shape))


def _find_min(indices, distances):
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


def compute_distance(prototypes, p_labels, data, d_labels, metric):
    distances = sp.spatial.distance.cdist(data, prototypes, metric)

    ii_same = np.transpose(np.array([d_labels == prototype_label for prototype_label in p_labels]))
    ii_diff = ~ii_same

    dist_same, i_dist_same = _find_min(ii_same, distances)
    dist_diff, i_dist_diff = _find_min(ii_diff, distances)

    return dist_same, dist_diff, i_dist_same, i_dist_diff
