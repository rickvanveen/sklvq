import numpy as np

from .test_common import check_distance
from .test_common import check_init_distance


from sklvq.models import GLVQ


def test_euclidean():
    check_init_distance("euclidean")

    data = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]], dtype="float64")
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    model = GLVQ(distance_type="euclidean")
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)

    check_distance(model._distance, data, model)

    # Check force_all_finite settings

    model = GLVQ(distance_type="euclidean", force_all_finite="allow-nan")
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)

    data[0, 0] = np.nan
    data[1, 0] = np.nan

    check_distance(model._distance, data, model)
