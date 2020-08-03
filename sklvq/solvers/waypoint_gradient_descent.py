from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

STATE_KEYS = ["variables", "nit", "fun", "nfun", "tfun", "njac", "tjac", "step_size"]


class WaypointGradientDescent(SolverBaseClass):
    """ WaypointGradientDescent

    Parameters
    ----------
    objective
    max_runs
    step_size
    loss
    gain
    k
    callback

    """
    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs=10,
        step_size=0.1,
        loss=2 / 3,
        gain=1.1,
        k=3,
        callback=None,
    ):
        super().__init__(objective)
        self.max_runs = max_runs
        self.step_size = step_size
        self.loss = loss
        self.gain = gain
        self.k = k
        self.callback = callback

    def solve(
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ) -> "LVQBaseClass":
        """

        Parameters
        ----------
        data
        labels
        model

        Returns
        -------

        """
        previous_objective_gradients = np.zeros(
            (self.k, model._to_variables(model._get_model_params()).size)
        )

        step_size = self.step_size
        # objective_gradient = None

        if self.callback is not None:
            variables = model._to_variables(model._get_model_params())
            cost = self.objective(variables, model, data, labels)
            state = self.create_state(
                STATE_KEYS,
                variables=variables,
                nit=0,
                nfun=cost,
                fun=cost
            )
            if self.callback(model, state):
                return model

        # Initial runs to get enough gradients to average.
        for i_run in range(0, self.k):
            shuffled_indices = shuffle(
                range(0, labels.size), random_state=model.random_state_
            )

            batch = data[shuffled_indices, :]
            batch_labels = labels[shuffled_indices]

            # Get model params variable shape (flattened)
            model_variables = model._to_variables(model._get_model_params())

            # Gradient in model_param form so can be prototypes or tuple(prototypes, omega)
            objective_gradient = model._to_params(
                self.objective.gradient(model_variables, model, batch, batch_labels)
            )

            # Normalize the gradient by gradient/norm(gradient)
            objective_gradient = model._normalize_params(objective_gradient)

            # Multiply params by step_size and transform to variables shape
            objective_gradient = model._to_variables(
                self.multiply_model_params(step_size, objective_gradient)
            )

            # Store the previous objective_gradients in variables shape
            previous_objective_gradients[np.mod(i_run, self.k), :] = objective_gradient

            # Subtract objective gradient of model params in variables form
            # and
            new_model_variables = model_variables - objective_gradient

            # Transform back to parameters form and update the model
            model._set_model_params(
                model._to_params(new_model_variables)
            )

            if self.callback is not None:
                cost = self.objective(new_model_variables, model, data, labels)
                state = self.create_state(
                    STATE_KEYS,
                    variables=new_model_variables,
                    nit=i_run + 1,
                    nfun=cost,
                    fun=cost,
                    njac=objective_gradient,  # scaled with the step_size
                    step_size=step_size,
                )
                if self.callback(model, state):
                    return model

        # The remainder of the runs
        for i_run in range(self.k, self.max_runs):
            shuffled_indices = shuffle(
                range(0, labels.size), random_state=model.random_state_
            )

            batch = data[shuffled_indices, :]
            batch_labels = labels[shuffled_indices]

            # Get model params variable shape (flattened)
            model_variables = model._to_variables(model._get_model_params())

            # Gradient in model_param form so can be prototypes or tuple(prototypes, omega)
            objective_gradient = model._to_params(
                self.objective.gradient(model_variables, model, batch, batch_labels)
            )

            # Normalize the gradient by gradient/norm(gradient)
            objective_gradient = model._normalize_params(objective_gradient)

            # Multiply params by step_size and transform to variables shape
            objective_gradient = model._to_variables(
                self.multiply_model_params(step_size, objective_gradient)
            )

            # Tentative update step cost
            tentative_objective_gradient = np.mean(
                previous_objective_gradients, axis=0
            )

            tentative_model_variables = model_variables - tentative_objective_gradient
            # tentative_model_variables = mean_previous_objective_gradients

            # Update the model using normalized update step
            new_model_variables = model_variables - objective_gradient

            # Compute cost of tentative update step
            tentative_cost = self.objective(
                tentative_model_variables, model, batch, batch_labels
            )  # Note: Objective updates the model to the tentative_model_params

            # New update step cost
            new_cost = self.objective(
                new_model_variables,
                model,
                data[shuffled_indices, :],
                labels[shuffled_indices],
            )  # Note: Objective updates the model to the new_model_params

            if tentative_cost < new_cost:
                model._set_model_params(model._to_params(tentative_model_variables))
                step_size = self.loss * step_size
                accepted_cost = tentative_cost
                accepted_gradient = tentative_objective_gradient
                accepted_variables = tentative_model_variables
            else:
                step_size = self.gain * step_size
                accepted_cost = new_cost
                accepted_gradient = objective_gradient
                accepted_variables = new_model_variables
                # We keep the model currently containing the new update -> because objective
                # updates the model and was called last...

            # Administration. Store the models parameters.
            previous_objective_gradients[np.mod(i_run, self.k), :] = accepted_gradient

            if self.callback is not None:
                state = self.create_state(
                    STATE_KEYS,
                    variables=accepted_variables,
                    nit=i_run + 1,
                    tfun=tentative_cost,
                    nfun=new_cost,
                    fun=accepted_cost,
                    # tjac=mean_previous_objective_gradients,
                    njac=objective_gradient,  # scaled with the step_size
                    step_size=step_size,
                )
                if self.callback(model, state):
                    return model

        return model
