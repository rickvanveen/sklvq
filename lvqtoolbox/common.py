# TODO: LVQClassifier - base class for LVQClassifier
# Abstract classes are not Pythonian apparently. However, common functionality which we always need can still be put in
# a common base class.

import numpy as np
import scipy as sp

from scipy.spatial.distance import cdist

def _conditional_mean(p_labels, data, d_labels):
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


def init_prototypes(p_labels, data, d_labels):
    return _conditional_mean(p_labels, data, d_labels)


def compute_distance(prototypes, p_labels, data, d_labels, metric):
    distances = sp.spatial.distance.cdist(data, prototypes, 'sqeuclidean')

    ii_same = np.transpose(np.array([d_labels == prototype_label for prototype_label in p_labels]))
    ii_diff = ~ii_same

    # TODO: Function
    dist_temp = np.where(ii_same, distances, np.inf)
    dist_same = dist_temp.min(axis=1)
    i_dist_same = dist_temp.argmin(axis=1)

    dist_temp = np.where(ii_diff, distances, np.inf)
    dist_diff = dist_temp.min(axis=1)
    i_dist_diff = dist_temp.argmin(axis=1)

    return dist_same, dist_diff, i_dist_same, i_dist_diff