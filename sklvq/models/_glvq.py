import numpy as np

from . import LVQBaseClass
from .. import distances, solvers
from ..objectives import GeneralizedLearningObjective
from ..solvers import SolverBaseClass

# Typing
from typing import Union

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

    This model optimizes the generalized learning objective introduced in [1]_.

    Parameters
    ----------
    distance_type : {"squared-euclidean", "euclidean"} or Class, default="squared-euclidean"
        The distance function. Can be one from the list above or a custom class.

    distance_params : dict, optional, default=None
        Parameters passed to init of distance class.

    activation_type : {"identity", "sigmoid", "soft+", "swish"} or Class, default="sigmoid"
        The activation function used in the objective function. Can be any of the activation
        function in the list or custom class.

    activation_params : dict, default=None
        Parameters passed to init of activation function. See the documentation of activation
        functions for function dependent parameters and defaults.

    discriminant_type : "relative-distance" or Class
        The discriminant function.

    discriminant_params : dict, default=None
        Parameters passed to init of discriminant callable

    solver_type : {"sgd", "wgd", "adam", "lbfgs", "bfgs"},
        The solver used for optimization

        - "sgd" or "steepest-gradient-descent"
            Refers to the stochastic and (mini) batch steepest descent optimizers.

        - "wgd" or "waypoint-gradient-descent"
            Implementation based on [2]_

        - "adam" or "adaptive-moment-estimation"
            Implementation based on description by [3]_

        - "bfgs" or "broyden-fletcher-goldfarb-shanno"
            Implementation from scipy package.

        - "lbfgs" or "limited-memory-bfgs"
            Implementation from scipy package.

    solver_params : dict, default=None
        Parameters passed to init of solvers. See the documentation of the solver
        functions for relevant parameters and defaults.

    initial_prototypes : "class-conditional-mean" or ndarray, default="class-conditional-mean"
        Default will initiate the prototypes to the class conditional mean with a small random
        offset. Custom numpy array can be passed to change the initial positions of the prototypes.

    prototypes_per_class : int or ndarray, optional, default=1
        Number of prototypes per class. Default will generate single prototype per class. In the
        case of unequal number of prototypes per class is preferable provide the labels as
        np.ndarray. Example prototypes_per_class = np.array([0, 0, 1, 2, 2, 2]) this will match
        with a total of 6 prototypes with first two class with index 0, then one with class index 1,
        and three with class index 2. Note: labels are indexes to classes\_ attribute.

    random_state : int, RandomState instance, default=None
        Determines random number generation. Used in random offset of prototypes and shuffling of
        the X in the solvers.

    force_all_finite : {True, "allow-nan"}, default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:

        - True: Force all values of array to be finite.
        - "allow-nan": accepts only np.nan and pd.NA values in array. Values cannot be infinite.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    prototypes_ : ndarray of shape (n_protoypes, n_features)
        Positions of the prototypes after fit(X, labels) has been called.

    prototypes_labels_ : ndarray of shape (n_prototypes)
        Labels for each prototypes. Labels are indexes to classes\_

    References
    ----------
    .. [1] Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
        Advances in Neural Network Information Processing Systems, 423–429, 1996.

    .. [2] Papari, G., and Bunte, K., and Biehl, M. (2011) "Waypoint averaging and step size
        control in learning by gradient descent" Mittweida Workshop on Computational
        Intelligence (MIWOCI) 2011.

    .. [3] LeKander, M., Biehl, M., & De Vries, H. (2017). "Empirical evaluation of gradient
        methods for matrix learning vector quantization." 12th International Workshop on
        Self-Organizing Maps and Learning Vector Quantization, Clustering and Data
        Visualization, WSOM 2017.
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
        prototype_params: dict = None,
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
            prototype_params,
            random_state,
            force_all_finite,
        )

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    def set_model_params(self, model_params: ModelParamsType) -> None:
        """
        Changes the model's internal parameters. Copies the values in model_params into
        self.prototypes_ therefor updating the variables_ array.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        self.set_prototypes(model_params)

    def get_model_params(self) -> ModelParamsType:
        """

        Returns
        -------
        ndarray
             Returns the prototypes as ndarray.

        """
        return self.get_prototypes()

    ###########################################################################################
    # Functions to transform the 1D variables array to model parameters and back
    ###########################################################################################

    def to_model_params(self, variables: np.ndarray) -> ModelParamsType:
        """
        Should create a view of the variables array in prototype shape.

        Parameters
        ----------
        variables : ndarray
            Single ndarray that stores the parameters in the order as given by the
            "to_variabes()" function

        Returns
        -------
        ndarray
            Returns the prototypes as ndarray.

        """
        return self.to_prototypes(variables)

    def to_prototypes(self, var_buffer: np.ndarray) -> np.ndarray:
        return var_buffer.reshape(self._prototypes_shape)

    ###########################################################################################
    # Solver Normalization functions
    ###########################################################################################

    def normalize_variables(self, var_buffer: np.ndarray) -> None:
        """

        Parameters
        ----------
        model_params : ndarray
            Model parameters as provided by get_model_params()

        Returns
        -------
        ndarray or tuple
            Same shape and size as input, but normalized. How to normalize depends on model
            implementation.

        """
        LVQBaseClass._normalize_prototypes(self.to_prototypes(var_buffer))

    ###########################################################################################
    # Solver helper functions
    ###########################################################################################

    def add_partial_gradient(self, gradient, partial_gradient, i_prototype) -> None:
        """

        Parameters
        ----------
        gradient
        partial_gradient
        i_prototype

        Returns
        -------

        """
        n_features = self.n_features_in_

        prots_view = self.to_prototypes(gradient)
        np.add(
            prots_view[i_prototype, :],
            partial_gradient[:n_features],
            out=prots_view[i_prototype, :],
        )

    def multiply_variables(
        self, step_size: Union[int, float], var_buffer: np.ndarray
    ) -> None:
        var_buffer *= step_size

    ###########################################################################################
    # Initialization required functions
    ###########################################################################################

    def _init_variables(self) -> None:
        self._variables = np.empty(self._prototypes_size, dtype="float64", order="C")

    def _check_model_params(self):
        self._check_prototype_params(**self.prototype_params)

    def _init_model_params(self, X, y) -> None:
        self._init_prototypes(X, y, **self.prototype_params)

    def _init_objective(self):
        self._objective = GeneralizedLearningObjective(
            self.activation_type,
            self.activation_params,
            self.discriminant_type,
            self.discriminant_params,
        )
