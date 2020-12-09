import numpy as np

from .test_common import DummyLVQ
from .test_common import check_distance
from .test_common import check_init_distance


def test_squared_euclidean():
    distance_class = check_init_distance("squared-euclidean")

    distfun = distance_class()

    data = np.array([[1, 2, 3], [-1, -2, -3]], dtype="float64")
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    model = DummyLVQ(p)

    check_distance(distfun, data, model)

    # Check force_all_finite settings
    distfun = distance_class(force_all_finite="allow-nan")

    data[0, 0] = np.nan
    data[1, 0] = np.nan

    model = DummyLVQ(p)
    check_distance(distfun, data, model)
