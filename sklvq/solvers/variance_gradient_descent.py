# Stochastic gradient descent
# vSGD maintains three exponential moving averages: of the gradient, g, of the Hadamard squared gradient, v,
# and of the Hesse diagonal, h
# Only hyperparameter C > 1, which does something that has to do with initialization.

from sklearn.utils import shuffle
import numpy as np

from . import SolverBaseClass
from sklvq.objectives import ObjectiveBaseClass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQClassifier

import numdifftools as nd


class VarianceGradientDescent(SolverBaseClass):

    def __init__(self, c=10, max_runs=20):
        self.c = c
        self.max_runs = max_runs

    def solve(self, data: np.ndarray, labels: np.ndarray, objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':

        # Administration
        variables_size = model.to_variables(model.get_model_params()).size

        shuffled_indices = shuffle(
            range(0, labels.size),
            random_state=model.random_state_)

        P = labels.size

        g = np.zeros(variables_size)
        v = np.zeros(variables_size)
        h = np.zeros(variables_size)

        hess_diagonal = nd.Hessdiag(objective)

        for i_sample in range(0, len(shuffled_indices)):
            # Get sample and its label
            sample = np.atleast_2d(
                data[shuffled_indices[i_sample], :])

            sample_label = np.atleast_1d(
                labels[shuffled_indices[i_sample]])

            # Get model params variable shape (flattened)
            model_variables = model.to_variables(
                model.get_model_params()
            )

            # Gradient in variables form
            objective_gradient = objective.gradient(
                model_variables,
                model,
                sample,
                sample_label
            )

            g += (1 / P) * objective_gradient
            v += (self.c / P) * objective_gradient ** 2
            h += (self.c / P) * hess_diagonal(model_variables, model, sample, sample_label)

        tau = P * np.ones(variables_size)

        for i_run in range(0, self.max_runs):
            # Randomize order of data
            shuffled_indices = shuffle(
                range(0, labels.size),
                random_state=model.random_state_)

            for i_sample in range(0, len(shuffled_indices)):
                # Get sample and its label
                sample = np.atleast_2d(
                    data[shuffled_indices[i_sample], :])

                sample_label = np.atleast_1d(
                    labels[shuffled_indices[i_sample]])

                # Get model params variable shape (flattened)
                model_variables = model.to_variables(
                    model.get_model_params()
                )

                # Gradient in variables form
                objective_gradient = objective.gradient(
                    model_variables,
                    model,
                    sample,
                    sample_label
                )

                inv_tau = 1 / tau
                g = (((1 - inv_tau) * g)
                     + (inv_tau * objective_gradient))

                v = (((1 - inv_tau) * v)
                     + (inv_tau * (objective_gradient ** 2)))

                h = (((1 - inv_tau) * h)
                     + (inv_tau * hess_diagonal(model_variables, model, sample, sample_label)))

                tau = (1 - (g**2) / v) * tau + 1
                eta = (g**2) / (v * h)

                model.set_model_params(
                    model.to_params(
                        model_variables - eta * objective_gradient
                    )
                )

        return model
