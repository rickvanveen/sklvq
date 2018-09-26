import numpy as np

from lvqtoolbox.distance import compute_distance


# GLVQ - mu(x) functions
def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


# derivative mu(x) in glvq paper, same and diff are relative to the currents prototype's label
def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff)**2


def relative_distance_difference_wrapper(prototypes, p_labels, data, d_labels, kwargs):
    return relative_distance_difference_cost(prototypes, p_labels, data, d_labels, **kwargs)

# GLVQ - Cost functions TODO: from scalefun and further kwargs...
def relative_distance_difference_cost(prototypes, p_labels,
                                      data, d_labels, scalefun, metricfun,
                                      scalefun_grad, metricfun_grad, scalefun_kwargs, metricfun_kwargs):
    num_features = data.shape[1]
    num_prototypes = p_labels.size

    prototypes = prototypes.reshape(p_labels.size, num_features)
    dist_same, dist_diff, i_dist_same, i_dist_diff = compute_distance(prototypes, p_labels, data, d_labels,
                                                                      metricfun, metricfun_kwargs)

    gradient = np.zeros(prototypes.shape)

    for i_prototype in range(0, num_prototypes):
        ii_same = i_prototype == i_dist_same
        ii_diff = i_prototype == i_dist_diff

        # f'(mu(x)) * (2 * d_2(x) / (d_1(x) + d_2(x))^2)
        relative_dist_same = (scalefun_grad(_relative_distance(dist_same[ii_same], dist_diff[ii_same]), **scalefun_kwargs) *
                             _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
        relative_dist_diff = (scalefun_grad(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff]), **scalefun_kwargs) *
                             _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))
        # -2 * (x - w)
        grad_same = metricfun_grad(data[ii_same, :], prototypes[i_prototype, :])
        grad_diff = metricfun_grad(data[ii_diff, :], prototypes[i_prototype, :])

        gradient[i_prototype, :] = (relative_dist_same.dot(grad_same) - relative_dist_diff.dot(grad_diff))

    return np.sum(scalefun(_relative_distance(dist_same, dist_diff), **scalefun_kwargs)), gradient.ravel()



