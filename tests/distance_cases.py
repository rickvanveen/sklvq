"""Lightweight model state for distance tests."""

from types import SimpleNamespace

import numpy as np

from sklvq.models import GMLVQ, LGMLVQ


def make_distance_model(prototypes, omega=None, localization=None):
    # Ordinary distances need only prototypes, the finite-value setting and a parameter getter.
    model = SimpleNamespace(
        prototypes_=np.array(prototypes, dtype=float),
        force_all_finite=True,
    )
    model.get_model_params = lambda: model.prototypes_

    # Assign consecutive prototypes to two classes, or one class for a single prototype.
    if localization is not None:
        model.relevance_localization = localization
        n_prototypes = len(model.prototypes_)
        model.prototypes_labels_ = np.repeat([0, 1], (n_prototypes + 1) // 2)[:n_prototypes]
        model._prototypes_shape = model.prototypes_.shape

    # Adaptive distances additionally consume relevance factors and their quadratic-form matrices.
    if omega is not None:
        model.omega_ = np.array(omega, dtype=float)
        model.get_model_params = lambda: (model.prototypes_, model.omega_)
        # Use the real relevance calculation without initializing or fitting a model.
        model._compute_lambda = LGMLVQ._compute_lambda if localization is not None else GMLVQ._compute_lambda

    return model
