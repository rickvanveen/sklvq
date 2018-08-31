# TODO: WRITE COMMENTS FOR THE FUNCTIONS

import numpy as np


def _conditional_mean(p_labels, data, d_labels):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


def init_prototypes(p_labels, data, d_labels, rng):
    """ Initializes the protoypes using the conditional mean and adds a small random value to break symmetry."""
    conditional_mean = _conditional_mean(p_labels, data, d_labels)
    return conditional_mean + (1e-4 * rng.uniform(-1, 1, conditional_mean.shape))


