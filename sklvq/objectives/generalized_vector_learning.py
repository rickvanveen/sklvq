from . import GeneralizedBaseObjective
import numpy as np


# Tightly coupled with GMLVQClassifier
class GeneralizedVectorLearning(GeneralizedBaseObjective):

    def __init__(self, activation=None, discriminant=None):
        super(GeneralizedVectorLearning, self).__init__(activation, discriminant)

    def _gradient(self, activation_gradient, discriminant_gradient, discriminant_score, data, model, i_prototype):
        # Should return the gradient with respect to all the variables of the model.
        # Using the notation from the original GLVQ paper (Sato and Yamada 1996)
        prototype_gradient = np.zeros(model.prototypes_.shape)

        # # TODO: Here also a function(Object) can be linked to deal with it...
        # # Assuming global relevance matrix TODO: support local per class and per prototype
        # if model.omega_type == 'local-per-class':
        #     pass # depends on prototypes class model.prototoypes_labels_[i_prototype]
        # elif model.omega_type == 'local-per-prototype':
        #     pass # Depends on prototype so same index as i_prototype
        # else:
        #     pass # global and easiest...
        # # omega_gradient = np.zeros(model.omega_.shape)

        activation_discriminant_gradient = (activation_gradient * discriminant_gradient)

        # Computes the following partial derivatives: ddi/dwi, with i = 1,2(depending on input)
        prototype_distance_gradient, omega_distance_gradient = model.distance_.gradient(data, model, i_prototype)

        # Finally: dS/dwi, with i = 1,2
        prototype_gradient[i_prototype, :] = (activation_discriminant_gradient.dot(prototype_distance_gradient)).squeeze()

        omega_gradient = activation_discriminant_gradient.dot(np.atleast_2d(omega_distance_gradient))

        # Here it is in 2d shape now we have to put it to the correct 1d, which is handled by the model.
        return model.to_variables(prototype_gradient, omega_gradient)
