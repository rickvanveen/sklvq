# it's a BGD
# 5 Hyperparameters or 4....: learning rates (prototypes, [omega]), the number of epochs the waypoints are averaged,
# loss < 1 and gain > 1 learning rate change factors for waypoint averaging and gradient steps...
# No update of learning steps... due to A normalized gradient I think
from copy import deepcopy

from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class WaypointGradientDescent(SolverBaseClass):

    def __init__(self, max_runs=10, step_size=0.2, loss=0.1, gain=1.1, k=3):
        self.max_runs = max_runs
        self.step_size = step_size
        self.loss = loss
        self.gain = gain
        self.k = k

    def solve(self, data: np.ndarray,
              labels: np.ndarray,
              objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':

        previous_objective_gradients = np.zeros((self.k, model.to_variables(model.get_model_params()).size))

        step_size = self.step_size
        # Initial runs to get enough gradients to average.
        for i_run in range(0, self.k):
            shuffled_indices = shuffle(
                range(0, labels.size),
                random_state=model.random_state_)

            batch = data[shuffled_indices, :]
            batch_labels = labels[shuffled_indices]

            # Get model params variable shape (flattened)
            model_variables = model.to_variables(
                model.get_model_params()
            )

            # Gradient in model_param form so can be prototypes or tuple(prototypes, omega)
            objective_gradient = model.to_params(
                objective.gradient(
                    model_variables,
                    model,
                    batch,
                    batch_labels
                )
            )

            # Normalize the gradient by gradient/norm(gradient)
            objective_gradient = model.normalize_params(objective_gradient)

            # Multiply params by step_size and transform to variables shape
            objective_gradient = model.to_variables(
                model.mul_params(
                    objective_gradient,
                    step_size
                )
            )

            # Store the previous objective_gradients in variables shape
            previous_objective_gradients[np.mod(i_run, self.k), :] = objective_gradient

            # Update the model
            model.set_model_params(
                model.to_params(
                    model_variables - objective_gradient
                )
            )

        # The remainder of the runs
        for i_run in range(self.k, self.max_runs):
            shuffled_indices = shuffle(
                range(0, labels.size),
                random_state=model.random_state_)

            batch = data[shuffled_indices, :]
            batch_labels = labels[shuffled_indices]

            # Get model params variable shape (flattened)
            model_variables = model.to_variables(
                model.get_model_params()
            )

            # Gradient in model_param form so can be prototypes or tuple(prototypes, omega)
            objective_gradient = model.to_params(
                objective.gradient(
                    model_variables,
                    model,
                    batch,
                    batch_labels
                )
            )

            # Normalize the gradient by gradient/norm(gradient)
            objective_gradient = model.normalize_params(objective_gradient)

            # Multiply params by step_size and transform to variables shape
            objective_gradient = model.to_variables(
                model.mul_params(
                    objective_gradient,
                    step_size
                )
            )

            # Tentative update step cost
            tentative_model_params = model.to_params(
                np.mean(previous_objective_gradients, axis=0)
            )

            # Update the model using normalized update step
            new_model_params = model.to_params(
                model_variables - objective_gradient
            )

            # Compute cost of tentative update step
            tentative_cost = objective(
                model.to_variables(tentative_model_params),
                model,
                batch,
                batch_labels
            )  # Note: Objective updates the model to the tentative_model_params

            # New update step cost
            new_cost = objective(
                model.to_variables(new_model_params),
                model,
                data[shuffled_indices, :],
                labels[shuffled_indices]
            )  # Note: Objective updates the model to the new_model_params

            if tentative_cost < new_cost:
                model.set_model_params(tentative_model_params)
                step_size = self.loss * step_size
            else:
                step_size = self.gain * step_size
                # We keep the model currently containing the new update

            # Administration. Store the models parameters.
            previous_objective_gradients[np.mod(i_run, self.k), :] = model.to_variables(model.get_model_params())

        return model
