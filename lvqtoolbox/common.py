# TODO: LVQClassifier - base class for LVQClassifier
# Abstract classes are not Pythonian apparently. However, common functionality which we always need can still be put in
# a common base class.

import numpy as np


def _conditional_mean(p_labels, data, d_labels):
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


def init_prototypes(p_labels, data, d_labels):
    return _conditional_mean(p_labels, data, d_labels)
