import numpy as np


def finite_difference(function, parameters):
    columns = []
    step = 1e-6

    # Perturb raw parameters so relevance normalization cannot change the derivative being checked.
    for parameter in parameters:
        for index in range(parameter.size):
            original = parameter.flat[index]
            try:
                parameter.flat[index] = original + step
                plus = function()
                parameter.flat[index] = original - step
                minus = function()
            finally:
                parameter.flat[index] = original
            columns.append((plus - minus) / (2 * step))
    return np.column_stack(columns)
