import numpy as np

from .test_common import check_distance
from .test_common import check_init_distance

from sklvq.models import GMLVQ


def test_adaptive_squared_euclidean():
    check_init_distance("adaptive-squared-euclidean")

    # Some X and prototypes
    data = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]], dtype="float64")
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])
    o = np.identity(data.shape[1])

    model = GMLVQ(distance_type="adaptive-squared-euclidean")
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o)

    check_distance(model._distance, data, model)

    # Rectangular omega
    model = GMLVQ(distance_type="adaptive-squared-euclidean", relevance_n_components=2)
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o[:2, :])

    check_distance(model._distance, data, model)

    # Check force_all_finite settings
    model = GMLVQ(
        distance_type="adaptive-squared-euclidean", force_all_finite="allow-nan"
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o)

    data[0, 0] = np.nan
    data[1, 0] = np.nan

    check_distance(model._distance, data, model)
