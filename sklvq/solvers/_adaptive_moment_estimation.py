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
    r"""Adaptive moment estimation (ADAM)

    Implementation and description inspired by `[1]`_.

    Adam maintains two moving averages of the gradient (:math:`m, v`), which get updated for
    every sample at each epoch/run until the maximum runs (``max_runs``) has been reached:

    .. math::
        \mathbf{m} &= \beta_1 \cdot \mathbf{m} + (1 - \beta_1) \cdot \nabla e_i(\theta) \\
        \mathbf{v} &= \beta_2 \cdot \mathbf{v} + (1 - \beta_2) \cdot [\nabla e_i(\theta)]^{\circ 2}.

    Since :math:`m`  and :math:`v` are initialized to zero vectors, they are biased towards zero.
    To counteract this, unbiased estimates :math:`\hat{m}` and :math:`\hat{v}` are computed:

    .. math::
        \hat{\mathbf{m}} &= \mathbf{m} / (1 - \beta^p_1) \\
        \hat{\mathbf{v}} &= \mathbf{v} / (1 - \beta^p_2),

    where :math:`p` is initially 0, but afterwards it's increased by 1 each time before
    selecting a  new random sample. The unbiased estimates of the average gradient are then used
    for the update step:

    .. math::
        \theta = \theta - \eta \cdot \hat{\mathbf{m}} \odot \hat{\mathbf{v}}^{\circ \frac{1}{2}},

    with :math:`\eta` the ``step_size``. Additionally,  ``beta1``, and ``beta2``,  can be chosen
    by the user.

    Note that :math:`\odot` denotes the elementwise (Hadamard) product and :math:`\mathbf{x}^{
    \circ y}` the elementwise power operation.

    Parameters
    ----------
    objective: ObjectiveBaseClass, required
        This is/should be set by the algorithm.
    max_runs: int
        Number of runs over all the X. Should be >= 1
    beta1: float
        Controls the decay rate of the moving average of the gradient. Should be < 1.0
        and > 0.
    beta2: float
        Controls the decay rate of the moving average of the squared gradient. Should be < 1.0
        and > 0.
    step_size: float
        The step size to control the learning rate.
    epsilon: float
        Small value to overcome zero division

    callback: callable
        Callable with signature callable(state). If the callable returns True the solver
        will stop (early). The state object contains the following information:

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
    _`[1]` LeKander, M., Biehl, M., & De Vries, H. (2017). "Empirical evaluation of gradient
    methods for matrix learning vector quantization." 12th International Workshop on
    Self-Organizing Maps and Learning Vector Quantization, Clustering and Data
    Visualization, WSOM 2017.
    """

    def __init__(
        self,
        objective: ObjectiveBaseClass,
        max_runs: int = 10,
        beta1: float = 0.9,
        beta2: float = 0.999,
        step_size: float = 0.001,
        epsilon: float = 1e-4,
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

        if beta1 <= 0 or beta1 > 1.0:
            raise ValueError(
                "{}:  Expected beta1 to be > 0 and <= 1.0 but got beta1 = {}".format(
                    type(self).__name__, beta1
                )
            )
        self.beta1 = beta1

        if beta2 <= 0 or beta2 > 1.0:
            raise ValueError(
                "{}:  Expected beta1 to be > 0 and <= 1.0 but got beta2 = {}".format(
                    type(self).__name__, beta2
                )
            )
        self.beta2 = beta2

        if np.any(step_size <= 0):
            raise ValueError(
                "{}:  Expected step_size to be > 0, but got step_size = {}".format(
                    type(self).__name__, step_size
                )
            )
        self.step_size = step_size

        if epsilon < 0:
            raise ValueError(
                "{}:  Expected epsilon to be > 0, but got epsilon = {}".format(
                    type(self).__name__, epsilon
                )
            )
        self.epsilon = epsilon

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
                variables=np.copy(model.get_variables()),
                nit="Initial",
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
                    variables=np.copy(model.get_variables()),
                    nit=i_run + 1,
                    fun=self.objective(model, data, labels),
                    m_hat=m_hat,
                    v_hat=v_hat,
                )
                if self.callback(state):
                    return
