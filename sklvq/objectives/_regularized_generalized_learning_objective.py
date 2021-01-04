import numpy as np

from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import GMLVQ, LGMLVQ


from . import GeneralizedLearningObjective


class RegularizedGeneralizedLearningObjective(GeneralizedLearningObjective):
    def __init__(
        self,
        activation_type: Union[str, type],
        activation_params: dict,
        discriminant_type: Union[str, type],
        discriminant_params: dict,
        regularization: Union[int, float] = 0,
    ):
        super(RegularizedGeneralizedLearningObjective, self).__init__(
            activation_type, activation_params, discriminant_type, discriminant_params
        )
        if isinstance(regularization, (int, float)):
            self.regularization = regularization
        else:
            raise ValueError(
                "{}:  Expected regularization to be of type int or float but got {} instead".format(
                    type(self).__name__, type(regularization).__name__
                )
            )

    def __call__(
        self,
        model: Union["GMLVQ", "LGMLVQ"],
        data: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:

        objective_value = super(RegularizedGeneralizedLearningObjective, self).__call__(
            model, data, labels
        )
        return objective_value - 1 / 2 * self.regularization * np.log(
            np.linalg.det(model.omega_.dot(model.omega_.T))
        )

    def gradient(
        self,
        model: Union["GMLVQ", "LGMLVQ"],
        data: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:

        gradient_buffer = super(RegularizedGeneralizedLearningObjective, self).gradient(
            model, data, labels
        )
        omega = model.to_omega(gradient_buffer)

        omega += self.regularization * 2 * np.linalg.pinv(model.omega_).T

        return gradient_buffer
