import numpy as np
from scipy.spatial.distance import cdist


# GLVQ - mu(x) functions
def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


# derivative mu(x) in glvq paper, same and diff are relative to the currents prototype's label
def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff)**2


# GLVQ - Cost functions
# TODO: Move to common, with metric and metric_args such
def _compute_distance(prototypes, p_labels, data, d_labels, metric):
    distances = cdist(data, prototypes, metric)

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


# TODO: f, and metric should be configurable. mu: relative distance also?
def relative_distance_difference_cost(prototypes, p_labels,
                                       data, d_labels,
                                       scalefun, metricfun, *args):
    # Prototypes are the x in for the to be optimized f(x, *args)
    prototypes = prototypes.reshape(p_labels.size, data.shape[1])
    dist_same, dist_diff, _, _ = _compute_distance(prototypes, p_labels, data, d_labels, metricfun)
    return np.sum(scalefun(_relative_distance(dist_same, dist_diff)))


# TODO: All shared arguments between cost and cost_grad functions args and the rest kwargs?
def relative_distance_difference_grad(prototypes, p_labels,
                                       data, d_labels,
                                       scalefun, metricfun,
                                       scalefun_grad, metricfun_grad):
    num_features = data.shape[1]
    num_prototypes = p_labels.size

    prototypes = prototypes.reshape([num_prototypes, num_features])
    dist_same, dist_diff, i_dist_same, i_dist_diff = _compute_distance(prototypes,
                                                                       p_labels, data,
                                                                       d_labels, metricfun)
    gradient = np.zeros(prototypes.shape)

    # TODO: REMOVE
    step_size = 0.05

    for i_prototype in range(0, num_prototypes):
        ii_same = i_prototype == i_dist_same
        ii_diff = i_prototype == i_dist_diff

        # f'(mu(x)) * (2 * d_2(x) / (d_1(x) + d_2(x))^2)
        relative_dist_same = (scalefun_grad(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) *
                             _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
        relative_dist_diff = (scalefun_grad(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff])) *
                             _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))
        # -2 * (x - w)
        grad_same = metricfun_grad(data[ii_same, :], prototypes[i_prototype, :])
        grad_diff = metricfun_grad(data[ii_diff, :], prototypes[i_prototype, :])

        gradient[i_prototype, :] = step_size * (relative_dist_same @ grad_same - relative_dist_diff @ grad_diff)

    return gradient.ravel()