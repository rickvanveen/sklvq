from typing import TYPE_CHECKING, Union

import numpy as np
from sklearn.utils import shuffle

from . import SolverBaseClass
from ._base import _update_state
from ..objectives import ObjectiveBaseClass

if TYPE_CHECKING:
    from sklvq.models import LVQBaseClass

STATE_KEYS = ["variables", "nit", "fun", "step_size"]


class SteepestGradientDescent(SolverBaseClass):
    r"""Steepest gradient descent (SGD)

    Implements the steepest gradient descent optimization method. Can perform stochastic,
    mini-batch and batch gradient descent by changing the batch_size. Implementation is
    inspired by the description given in `[1]`_.

    The algorithm performs the following update of the model parameters (:math:`\mathbf{\theta}`) per
    batch. This process is repeated multiple times (per step) when the ``batch_size`` (:math:`M`)
    is smaller than the total number of samples in the data.

    .. math::
        \mathbf{\theta} = \mathbf{\theta}  - \eta(t) \cdot \sum_i^M \nabla  e_i(\mathbf{\theta}),

    with :math:`\nabla e_i(\mathbf{\theta})` the gradient of the objective function with respect to
    a sample given the current model parameters :math:`\mathbf{\theta}`, and :math:`\eta(t)` the step
    size at step :math:`t`, which is changed using a simple annealing function:

    .. math::
        \eta(t) = \frac{\eta_{init}} {(1 + \frac{t}{t_{max}})},

    with :math:`t_{max}` given by the ``max_runs`` parameter and :math:`\eta_{init}` by the
    ``step_size`` parameter.


    Parameters
    ----------
    objective: ObjectiveBaseClass, required
        This is set by the algorithm. See :class:`sklvq.models.GLVQ`, :class:`sklvq.models.GMLVQ`,
        and :class:`sklvq.models.LGMLVQ`.

    max_runs: int
        Maximum number of runs/epochs that will be computed. Should be >= 1. Early stopping can
        be implemented by providing a ``callback`` function that returns True when the solver should
        stop.

    batch_size: int
        Controls the batch size and accepts a value >= 0. The value indicates the number of
        samples considered to be in the batch. A stochastic gradient descent corresponds with a
        ``batch_size`` of 1. For Batch gradient descent 0 can be used to indicate to use all the
        samples. Any value > 1 < n_samples can be considered as a mini-batch gradient descent.

        If batches can not properly  be divided in batches with the specified size the last
        batch might contain less than the specified number of samples.

        The data is always shuffled before it is split into batches.

    step_size: float or ndarray
        The step size to control the learning rate of the model parameters. If the same step size
        should be used for all parameters (e.g., prototypes and omega) then a single float is
        sufficient. If separate initial step sizes should be used per model parameter then this
        should be specified by using a numpy array.

    callback: callable
        Callable with signature callable(state). If the callable returns True the solver
        will stop even if ``max_runs`` is not reached yet. The state object contains the following:

        - "variables"
            Concatenated 1D ndarray of the model's parameters
        - "nit"
            The current iteration counter
        - "fun"
            The objective cost
        - "step_size"
            The current step_size(s)

    References
    ----------
    _`[1]` LeKander, M., Biehl, M., & De Vries, H. (2017). "Empirical evaluation of gradient
    methods for matrix learning vector quantization." 12th International Workshop on
    Self-Organizing Maps and Learning Vector Quantization, Clustering and Data
    Visualization, WSOM 2017.
    """

    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs: int = 10,
        batch_size: int = 1,
        step_size: Union[float, np.ndarray] = 0.1,
        callback: callable = None,
    ):
        super().__init__(objective)
        if max_runs <= 0:
            raise ValueError(
                "{}:  Expected max_runs to be > 0, but got max_runs = {}".format(
                    type(self).__name__, max_runs
                )
            )
        self.max_runs = max_runs

        if batch_size < 0:
            raise ValueError(
                "{}:  Expected batch_size to be >= 0, but got batch_size = {}".format(
                    type(self).__name__, batch_size
                )
            )
        self.batch_size = batch_size

        if np.any(step_size <= 0):
            raise ValueError(
                "{}:  Expected step_size to be > 0, but got step_size = {}".format(
                    type(self).__name__, step_size
                )
            )
        self.step_size = step_size

        if callback is not None:
            if not callable(callback):
                raise ValueError(
                    "{}:  callback is not callable.".format(type(self).__name__)
                )
        self.callback = callback

    def solve(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        model: "LVQBaseClass",
    ):
        """Solve function that gets called by the fit method of the models.

        Performs the steps of the steepest gradient descent optimization method.

        Parameters
        ----------
        data : ndarray of shape (n_samples, n_features)
            The data.

        labels : ndarray of size (n_samples)
            The labels of the samples in the data.

        model : LVQBaseClass
            The initial model that will also hold the final result
        """

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

        if batch_size > data.shape[0]:
            raise ValueError("Provided batch_size is invalid.")

        for i_run in range(0, self.max_runs):
            # Randomize order of samples
            shuffled_indices = shuffle(
                np.array(range(0, labels.size)), random_state=model.random_state_
            )

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
                # Creating views when batch_size is 1 does not do much for speed.
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

            if self.callback is not None:
                state = _update_state(
                    STATE_KEYS,
                    variables=np.copy(model.get_variables()),
                    nit=i_run + 1,
                    fun=self.objective(model, data, labels),
                    step_size=step_size,
                )
                if self.callback(state):
                    return
