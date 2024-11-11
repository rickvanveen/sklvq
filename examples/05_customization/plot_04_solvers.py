"""
=======
Solvers
=======
"""

from typing import TYPE_CHECKING

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from sklvq import GLVQ
from sklvq.objectives import ObjectiveBaseClass
from sklvq.solvers import SolverBaseClass
from sklvq.solvers._base import _update_state

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

STATE_KEYS = ["variables", "nit", "fun", "step_size"]

###############################################################################
# The sklvq package contains a number of different solvers.  Please see the API reference under
# Documentation for the full list.


class CustomSteepestGradientDescent(SolverBaseClass):
    def __init__(
        self,
        # init requires the objective instance to be given when  initialized. It will be passed
        # to the (super) solver base class.
        objective: ObjectiveBaseClass,
        max_runs: int = 10,
        batch_size: int = 1,
        step_size: float = 0.1,
        callback: callable = None,
    ):
        super().__init__(objective)
        # In the actual implementation checks can be done to ensure proper values for the
        # parameters of the solver (as is done in the actual code).
        self.max_runs = max_runs
        self.batch_size = batch_size
        self.step_size = step_size
        self.callback = callback

    def solve(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model: "LVQBaseClass",
    ):
        # Calls the callback function is provided with the initial values.
        if self.callback is not None:
            state = _update_state(
                STATE_KEYS,
                variables=np.copy(model.get_variables()),
                nit="Initial",
                fun=self.objective(model, data, labels),
            )
            if self.callback(state):
                return

        batch_size = self.batch_size

        # These checks cannot be done in init because data is not available at that moment.
        if batch_size > data.shape[0]:
            raise ValueError("Provided batch_size is invalid.")

        if batch_size <= 0:
            batch_size = data.shape[0]

        for i_run in range(self.max_runs):
            # Randomize order of samples
            shuffled_indices = shuffle(np.array(range(labels.size)), random_state=model.random_state_)

            # Divide the shuffled indices into batches (not necessarily equal size,
            # see documentation of numpy.array_split).
            batches = np.array_split(
                shuffled_indices,
                list(range(batch_size, labels.size, batch_size)),
                axis=0,
            )

            # Update step size using a simple annealing strategy
            step_size = self.step_size / (1 + i_run / self.max_runs)

            for i_batch in batches:
                # Select the data
                batch = data[i_batch, :]
                batch_labels = labels[i_batch]

                # Compute objective gradient
                objective_gradient = self.objective.gradient(model, batch, batch_labels)

                # Multiply each param by its given step_size
                model.mul_step_size(step_size, objective_gradient)

                # Update the model by subtracting the objective-gradient (descent) from the
                # current models variables, e.g., (prototypes, omega) in case of GMLVQ
                model.set_variables(
                    np.subtract(  # returns out=objective_gradient
                        model.get_variables(),
                        objective_gradient,
                        out=objective_gradient,
                    )
                )

            # Call the callback function if provided with updated values.
            if self.callback is not None:
                state = _update_state(
                    STATE_KEYS,
                    variables=np.copy(model.get_variables()),
                    nit=i_run + 1,
                    fun=self.objective(model, data, labels),
                    step_size=step_size,
                )
                # Simply return (stop the solver process) when callback returns true.
                if self.callback(state):
                    return


###############################################################################
# The CustomSteepestGradientDescent above, accompanied with some tests and documentation, would
# make a great addition to the sklvq package. However, it can also directly be passed to the
# algorithm. Some other solvers might require more functionality not supported by the models,
# this can be added dynamically to the model instances or by extending the required model and
# creating a custom model class.

data, labels = load_iris(return_X_y=True)

model = GLVQ(
    solver_type=CustomSteepestGradientDescent,
    distance_type="squared-euclidean",
    activation_type="sigmoid",
    activation_params={"beta": 2},
)

model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
