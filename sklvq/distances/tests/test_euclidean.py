import numpy as np

from .test_common import check_init_distance
from .test_common import check_distance
from .test_common import DummyLVQ


def test_euclidean():
    distance_class = check_init_distance("euclidean")

    distfun = distance_class()

    data = np.array([[1, 2, 3], [-1, -2, -3]])
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    model = DummyLVQ(p)

    check_distance(distfun, data, model)

    # Check force_all_finite settings