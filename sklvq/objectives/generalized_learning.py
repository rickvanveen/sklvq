from . import GeneralizedBaseObjective

import numpy as np


# TODO: regularization?

class GeneralizedLearning(GeneralizedBaseObjective):
    def __init__(self, activation=None, discriminant=None):
        super(GeneralizedLearning, self).__init__(activation, discriminant)

    # d1, d2, u(d1, d2), x, (w_i, [omega]), i's)
    def _gradient(self, activation_gradient, discriminant_gradient, discriminant_score, data, model, i_prototype):
        # Using the notation from the original GLVQ paper (Sato and Yamada 1996)
        prototype_gradient = np.zeros(model.prototypes_.shape)

        # Computes the following partial derivatives: ddi/dwi, with i = 1,2(depending on input)
        distance_gradient = model.distance_.gradient(data, model, i_prototype)

        # Finally: dS/dwi, with i = 1,2
        prototype_gradient[i_prototype, :] = ((activation_gradient * discriminant_gradient).dot(distance_gradient)).squeeze()

        # Here it is in 2d shape now we have to put it to the correct 1d, which is handled by the model.
        return model.to_variables(prototype_gradient)
