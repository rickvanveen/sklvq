from scipy.spatial.distance import cdist
import numpy as np


# TODO: Argument could just be the LVQ model object and use the relevant properties
def std_costfun(prototypes, prototype_labels, data, data_labels, metric):
    # Prototypes are the x in for the to be optimized f(x, *args)
    prototypes = prototypes.reshape([prototype_labels.size, data.shape[1]])

    distances = cdist(prototypes, data, metric)

    ii_same = np.array([data_labels == prototype_label for prototype_label in prototype_labels])
    ii_diff = ~ii_same

    distance_same = np.where(ii_same, distances, np.inf).min(axis=1)
    distance_diff = np.where(ii_diff, distances, np.inf).min(axis=1)

    distance_same_plus_diff = distance_same + distance_diff
    return np.sum((distance_same - distance_diff) / distance_same_plus_diff)


# TODO: Gradient function of the std_cost function... depending on how the minimizers work.
# def std_costfun_grad(modelObject, dataObject):
# TODO: PrototypeClass?

