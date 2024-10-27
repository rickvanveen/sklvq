import numpy as np

from .test_common import check_distance
from .test_common import check_init_distance

from sklvq.models import LGMLVQ


def test_adaptive_squared_euclidean():
    check_init_distance("local-adaptive-squared-euclidean")

    # Some X and prototypes
    data = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]], dtype="float64")
    p = np.array([[1, 2, 3], [-1, -2, -3], [0, 0, 0]])

    shape = (data.shape[1], data.shape[1])
    o = np.array([np.eye(*shape) for _ in range(p.shape[0])])

    model = LGMLVQ(
        distance_type="local-adaptive-squared-euclidean",
        relevance_localization="prototypes",
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o)

    check_distance(model._distance, data, model)

    model = LGMLVQ(
        distance_type="local-adaptive-squared-euclidean", relevance_localization="class"
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o)

    check_distance(model._distance, data, model)

    # Rectangular omega
    orec = np.array([np.eye(*shape)[:2, :] for _ in range(p.shape[0])])
    model = LGMLVQ(
        distance_type="local-adaptive-squared-euclidean",
        relevance_localization="prototypes",
        relevance_n_components=2,
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(orec)

    check_distance(model._distance, data, model)

    model = LGMLVQ(
        distance_type="local-adaptive-squared-euclidean",
        relevance_localization="class",
        relevance_n_components=2,
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(orec)

    check_distance(model._distance, data, model)

    # Check force_all_finite settings
    model = LGMLVQ(
        distance_type="local-adaptive-squared-euclidean",
        relevance_localization="prototypes",
        force_all_finite="allow-nan",
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o)

    nan_data = np.copy(data)
    nan_data[0, 0] = np.nan
    nan_data[1, 0] = np.nan

    check_distance(model._distance, nan_data, model)

    model = LGMLVQ(
        distance_type="local-adaptive-squared-euclidean",
        relevance_localization="class",
        force_all_finite="allow-nan",
    )
    model.fit(data, np.array([0, 1, 2]))
    model.set_prototypes(p)
    model.set_omega(o)

    check_distance(model._distance, nan_data, model)
