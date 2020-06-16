# Adam
# Stochastic gradient descent based
# Maintains two moving averages for the gradient m and the hadamard squared gradient v
# Beta1 and beta2 control the decay rate of these averages. (hyperparameters)
# Eta step size (in total 3 hyperparamters)
from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass


class AdaptiveMomentEstimation(SolverBaseClass):
    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs=20,
        beta1=0.9,
        beta2=0.999,
        step_size=0.001,
        epsilon=1e-4,
        callback=None
    ):
        super().__init__(objective)
        self.max_runs = max_runs
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_size = step_size
        self.epsilon = epsilon
        self.callback = callback

    def solve(
        self, data: np.ndarray, labels: np.ndarray, model: "LVQBaseClass",
    ) -> "LVQCLassifier":

        # Administration
        variables_size = model.to_variables(model.get_model_params()).size

        # Init/allocation of moving averages (m and v in literature)
        m = np.zeros(variables_size)
        v = np.zeros(variables_size)
        p = 0

        for i_run in range(0, self.max_runs):
            # Randomize order of data
            shuffled_indices = shuffle(
                range(0, labels.size), random_state=model.random_state_
            )

            for i_sample in range(0, len(shuffled_indices)):

                # Update power
                p += 1

                # Get sample and its label
                sample = np.atleast_2d(data[shuffled_indices[i_sample], :])

                sample_label = np.atleast_1d(labels[shuffled_indices[i_sample]])

                # Get model params variable shape (flattened)
                model_variables = model.to_variables(model.get_model_params())

                # Gradient in variables form
                objective_gradient = self.objective.gradient(
                    model_variables, model, sample, sample_label
                )

                # Update biased (init 0) moving gradient averages m and v.
                m = (self.beta1 * m) + ((1 - self.beta1) * objective_gradient)

                v = (self.beta2 * v) + ((1 - self.beta2) * objective_gradient ** 2)

                # Update unbiased moving gradient averages
                m_hat = m / (1 - self.beta1 ** p)

                v_hat = v / (1 - self.beta2 ** p)

                objective_gradient = (
                    self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
                )

                model.set_model_params(
                    model.to_params(model_variables - objective_gradient)
                )

            if self.callback is not None:
                if self.callback(data, labels, model):
                    return model

        return model
