# Typing
from typing import Union

import numpy as np

from . import LVQBaseClass
from ..objectives import GeneralizedLearningObjective

ModelParamsType = np.ndarray

DISTANCES = [
    "euclidean",
    "squared-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


class GLVQ(LVQBaseClass):
    r"""Generalized Learning Vector Quantization

    This model uses the :class:`sklvq.objectives.GeneralizedLearningObjective` as its objective
    function `[1]`_.

    Parameters
    ----------
    distance_type : {"squared-euclidean", "euclidean"} or Class, default="squared-euclidean"
        The distance function. Can be one from the following list or a custom class:

        - "squared-euclidean"
            See :class:`sklvq.distances.SquaredEuclidean`
        - "euclidean"
            See :class:`sklvq.distances.Euclidean`

    distance_params : dict, optional, default=None
        Parameters passed to init of distance class.

    activation_type : {"identity", "sigmoid", "soft+", "swish"} or Class, default="sigmoid"
        The activation function used in the objective function. Can be any of the activation
        function in the list or custom class.

        - "identity"
            See :class:`sklvq.activations.Identity`
        - "sigmoid"
            See :class:`sklvq.activations.Sigmoid`
        - "soft+"
            See :class:`sklvq.activations.SoftPlus`
        - "swish"
            See :class:`sklvq.activations.Swish`

    activation_params : dict, default=None
        Parameters passed to init of activation function. See the documentation of the activation
        functions for parameters and defaults.

    discriminant_type : "relative-distance" or Class
        The discriminant function.  Note that different discriminant type may require to rewrite
        the ``decision_function`` and ``predict_proba`` methods.

        - "relative-distance"
            See :class:`sklvq.discriminants.RelativeDistance`

    discriminant_params : dict, default=None
        Parameters passed to init of discriminant callable. See the documentation of the
        discriminant functions for parameters and defaults.

    solver_type : {"sgd", "wgd", "adam", "lbfgs", "bfgs"},
        The solver used for optimization

        - "sgd" or "steepest-gradient-descent"
            See :class:`sklvq.solvers.SteepestGradientDescent`.
        - "wgd" or "waypoint-gradient-descent"
            See :class:`sklvq.solvers.WaypointGradientDescent`.
        - "adam" or "adaptive-moment-estimation"
            See :class:`sklvq.solvers.AdaptiveMomentEstimation`.
        - "bfgs" or "broyden-fletcher-goldfarb-shanno"
            See :class:`sklvq.solvers.BroydenFletcherGoldfarbShanno`
        - "lbfgs" or "limited-memory-bfgs"
            See :class:`skvlq.solvers.LimitedMemoryBfgs`

    solver_params : dict, default=None
        Parameters passed to init of solvers. See the documentation of the solvers relevant
        parameters and defaults.

    prototype_init: "class-conditional-mean" or ndarray, default="class-conditional-mean"
        Default will initiate the prototypes to the class conditional mean with a small random
        offset. Custom numpy array can be passed to change the initial positions of the prototypes.

    prototype_n_per_class: int or np.ndarray, optional, default=1
        Default will generate single prototype per class. In the case of unequal number of
        prototypes per class is needed, provide this as np.ndarray. For example,
        prototype_n_per_class = np.array([1, 6, 3]) this will result in one prototype for the first class,
        six for the second, and three for the third. Note that the order needs to be the same as the on in the
        classes\_ attribute, which is equal to calling np.unique(labels).

    random_state : int, RandomState instance, default=None
        Set the random number generation for reproducibility purposes. Used in random offset of prototypes and
        shuffling of the data in the solvers.

    force_all_finite : {True, "allow-nan"}, default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:

        - True: Force all values of array to be finite.
        - "allow-nan": accepts only np.nan and pd.NA values in array. Values cannot be infinite.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The original and unique labels found in the data.

    prototypes_ : ndarray of shape (n_protoypes, n_features)
        Positions of the prototypes after fit(X, labels) has been called.

    prototypes_labels_ : ndarray of shape (n_prototypes)
        Labels for each prototypes. Labels are indexes to ``classes_``

    References
    ----------
    _`[1]` Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
    Advances in Neural Network Information Processing Systems, 423–429, 1996.
    """

    classes_: np.ndarray
    prototypes_: np.ndarray
    prototypes_labels_: np.ndarray

    def __init__(
        self,
        distance_type: Union[str, type] = "squared-euclidean",
        distance_params: dict = None,
        activation_type: Union[str, type] = "sigmoid",
        activation_params: dict = None,
        discriminant_type: Union[str, type] = "relative-distance",
        discriminant_params: dict = None,
        solver_type: Union[str, type] = "steepest-gradient-descent",
        solver_params: dict = None,
        prototype_init: str = "class-conditional-mean",
        prototype_n_per_class: Union[int, np.ndarray] = 1,
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, bool] = True,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params

        super(GLVQ, self).__init__(
            distance_type,
            distance_params,
            DISTANCES,
            solver_type,
            solver_params,
            SOLVERS,
            prototype_init,
            prototype_n_per_class,
            random_state,
            force_all_finite,
        )

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    def get_model_params(self) -> ModelParamsType:
        """
        Returns a view of all model parameters, which are only the prototypes.

        Returns
        -------
        ndarray
             Returns a view of the prototypes as ndarray.
        """
        return self.get_prototypes()

    def set_model_params(self, new_model_params: ModelParamsType) -> None:
        """
        Changes the model's internal parameters. Copies the values of model_params into
        ``self.prototypes_`` therefor updating the ``self.variables_`` array.

        Parameters
        ----------
        new_model_params : ndarray of shape (n_prototypes, n_features)
            In the   case the prototypes.
        """
        self.set_prototypes(new_model_params)

    ###########################################################################################
    # Functions to transform the 1D variables array to model parameters and back
    ###########################################################################################

    def to_model_params_view(self, var_buffer: np.ndarray) -> ModelParamsType:
        """
        Should create a view of the variables array in prototype shape.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        ndarray
            Returns the prototypes as ndarray.
        """
        return self.to_prototypes_view(var_buffer)

    def to_prototypes_view(self, var_buffer: np.ndarray) -> np.ndarray:
        """
        Returns the prototypes into the provided var_buffer. I.e., it selects/views the
        appropriate part of memory and reshapes it.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        ndarray of shape (n_prototypes, n_features)
            Prototype view into the var_buffer.
        """
        return var_buffer.reshape(self._prototypes_shape)

    ###########################################################################################
    # Solver Normalization functions
    ###########################################################################################

    def normalize_variables(self, var_buffer: np.ndarray) -> None:
        r"""
        Modifies the var_buffer as if it was the variables array provided
        by ``get_variables()``. As variables only contain prototypes it will now  contain the
        normalized prototypes.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        ndarray
            Same shape and size as input, but normalized.
        """
        LVQBaseClass._normalize_prototypes(self.to_prototypes_view(var_buffer))

    ###########################################################################################
    # Solver helper functions
    ###########################################################################################

    def add_partial_gradient(self, gradient, partial_gradient, i_prototype) -> None:
        """
        Adds the partial gradient to the correct part of the gradient, which  depends on
        ``i_prototype``.

        Parameters
        ----------
        gradient : ndarray
            Same shape as the ``get_variables()`` would return.

        partial_gradient : ndarray
            1d array containing the partial gradient.

        i_prototype : int
            The index of the prototype to which the partial gradient was  computed.
        """
        n_features = self.n_features_in_

        prots_view = self.to_prototypes_view(gradient)
        np.add(
            prots_view[i_prototype, :],
            partial_gradient[:n_features],
            out=prots_view[i_prototype, :],
        )

    def mul_step_size(self, step_size: Union[int, float], gradient: np.ndarray) -> None:
        """
        As GLVQ only has prototypes that are optimized the step_size should be a single float
        and can just be used to multiply the gradient inplace.

        Parameters
        ----------
        step_size : float or ndarray
            The scalar or list of values containing the step sizes.
        gradient : ndarray
            Same shape as the ``get_variables()`` would return.
        """
        gradient *= step_size

    ###########################################################################################
    # Initialization required functions
    ###########################################################################################

    def _init_variables(self) -> None:
        self._variables = np.empty(self._prototypes_size, dtype="float64", order="C")

    def _check_model_params(self):
        self._check_prototype_params()

    def _init_model_params(self, X, y) -> None:
        self._init_prototypes(X, y)

    def _init_objective(self):
        self._objective = GeneralizedLearningObjective(
            self.activation_type,
            self.activation_params,
            self.discriminant_type,
            self.discriminant_params,
        )
