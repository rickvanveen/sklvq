import numpy as np
from sklearn.utils import shuffle

from . import SolverBaseClass
from ..objectives import ObjectiveBaseClass
from ._base import _update_state

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

STATE_KEYS = ["variables", "nit", "fun", "m_hat", "v_hat"]


class AdaptiveMomentEstimation(SolverBaseClass):
    """AdaptiveMomentEstimation

    Implementation based on description given in [1]_.

    Parameters
    ----------
    objective: ObjectiveBaseClass, required
        This is/should be set by the algorithm.
    max_runs: int
        Number of runs over all the X. Should be >= 1
    beta1: float
        Controls the decay rate of the moving average of the gradient. Should be less than 1.0
        and greater than 0.
    beta2: float
        Controls the decay rate of the moving average of the squared gradient. Should be less
        than 1.0 and greater than 0.
    step_size: float
        The step size to control the learning rate.
    epsilon: float
        Small value to overcome zero division

    callback: callable
        Callable with signature callable(state). If the callable returns True the solver
        will stop (early). The state object contains the following.

        - "variables"
            Concatenated 1D ndarray of the model's parameters
        - "nit"
            The current iteration counter
        - "fun"
            The objective cost
        - "m_hat"
            Unbiased moving average of the gradient
        - "v_hat"
            Unbiased moving average of the Hadamard squared gradient

    References
    ----------
    .. [1] LeKander, M., Biehl, M., & De Vries, H. (2017). "Empirical evaluation of gradient
        methods for matrix learning vector quantization." 12th International Workshop on
        Self-Organizing Maps and Learning Vector Quantization, Clustering and Data
        Visualization, WSOM 2017.

    """

    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs: int = 20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        step_size: float = 0.001,
        epsilon: float = 1e-4,
        callback: callable = None,
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
    ):
        """

        Parameters
        ----------
        data : ndarray of shape (number of observations, number of dimensions)
        labels : ndarray of size (number of observations)
        model : LVQBaseClass
            The initial model that will be changed and holds the results at the end

        """

        # Administration
        variables = model.get_variables()
        variables_size = variables.size

        # Init/allocation of moving averages (m and v in literature)
        m = np.zeros(variables_size)
        v = np.zeros(variables_size)
        p = 0

        if self.callback is not None:
            state = _update_state(
                STATE_KEYS,
                variables=variables,
                nit=0,
                fun=self.objective(model, data, labels),
            )
            if self.callback(state):
                return

        for i_run in range(0, self.max_runs):
            # Randomize order of X
            shuffled_indices = shuffle(
                range(0, labels.size), random_state=model.random_state_
            )

            shuffled_data = data[shuffled_indices, np.newaxis, :]
            shuffled_labels = labels[shuffled_indices, np.newaxis]

            for i_sample, (sample, sample_label) in enumerate(
                zip(shuffled_data, shuffled_labels)
            ):
                # Update power
                p += 1

                objective_gradient = self.objective.gradient(
                    model, sample, sample_label
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

                model.set_variables(
                    np.subtract(  # returns out=objective_gradient
                        model.get_variables(),
                        objective_gradient,
                        out=objective_gradient,
                    )
                )

            if self.callback is not None:
                state = _update_state(
                    STATE_KEYS,
                    variables=model.get_variables(),
                    nit=i_run + 1,
                    fun=self.objective(model, data, labels),
                    m_hat=m_hat,
                    v_hat=v_hat,
                )
                if self.callback(state):
                    return
