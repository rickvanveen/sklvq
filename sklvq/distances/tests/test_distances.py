import pytest
import numpy as np
import pandas as pd

from sklvq import distances
from sklvq import GLVQClassifier

from sklvq.distances.cumulative_residual_entropy import CumulativeResidualEntropy
from sklvq.distances.cumulative_residual_entropy import _cre


def test_cumulative_residual_entropy():
    n_samples = 1
    n_dims = 10
    n_prototypes = 4

    random = np.random.RandomState(seed=31415)

    example_data = random.rand(10000).reshape(100, 100)
    # example_data = np.atleast_2d(np.linspace(1,10, 100))

    data = example_data[:, 20]
    lambdas = np.linspace(np.min(data), np.max(data), data.size)
    print("\n")
    print(_cre(data, lambdas))

    # example_prototypes = np.atleast_2d([2,2,2,2])
    # example_prototypes = np.random.rand(n_prototypes * n_dims).reshape(n_prototypes, n_dims)
    # example_prototypes = example_data

    # This looks weird but it's how it works
    # model = GLVQClassifier()
    # model.prototypes_ = example_prototypes

    # distance = CumulativeResidualEntropy()
    # dist_fun = distances.grab('cumulative-residual-entropy', None)
    # dists = dist_fun(example_data, model)


    # print(dists)
