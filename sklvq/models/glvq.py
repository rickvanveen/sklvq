from . import LVQBaseClass
import numpy as np

# Can be switched out by parameters to the models.
from sklvq import distances, solvers

# Typing
from typing import Union
from typing import Dict


from sklvq.solvers import SolverBaseClass

# Cannot be switched out by parameters to the models.
from sklvq.objectives import GeneralizedLearningObjective

ModelParamsType = np.ndarray

DISTANCE_FUNCTIONS = [
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

    This model optimizes the generalized learning objective introduced in [1]_. The 

    Parameters
    ----------
    distance_type : {"squared-euclidean", "euclidean"} or Class, default="squared-euclidean"
        The distance function. Can be one from the list above or a custom class.

    distance_params : Dict, optional, default=None
        Parameters passed to init of distance class.

    activation_type : {"identity", "sigmoid", "soft+", "swish"} or Class, default="sigmoid"
        The activation function used in the objective function. Can be any of the activation
        function in the list or custom class.

    activation_params : Dict, default=None
        Parameters passed to init of activation function. See the documentation of activation
        functions for function dependent parameters and defaults.

    discriminant_type : "relative-distance" or Class
        The discriminant function.

    discriminant_params : Dict, default=None
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

    solver_params : Dict, default=None
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
        the data in the solvers.

    force_all_finite : {True, "allow-nan"}, default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:

        - True: Force all values of array to be finite.
        - "allow-nan": accepts only np.nan and pd.NA values in array. Values cannot be infinite.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    prototypes_ : ndarray of shape (n_protoypes, n_features)
        Positions of the prototypes after fit(data, labels) has been called.

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
        distance_params: Dict = None,
        activation_type: Union[str, type] = "sigmoid",
        activation_params: Dict = None,
        discriminant_type: Union[str, type] = "relative-distance",
        discriminant_params: Dict = None,
        solver_type: Union[str, type] = "steepest-gradient-descent",
        solver_params: Dict = None,
        initial_prototypes: Union[str, np.ndarray] = "class-conditional-mean",
        prototypes_per_class: Union[int, np.ndarray] = 1,
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
            solver_type,
            solver_params,
            initial_prototypes,
            prototypes_per_class,
            random_state,
            force_all_finite,
        )

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    def _set_model_params(self, model_params: ModelParamsType) -> None:
        """
        Changes the model's internal parameters.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        self.prototypes_ = model_params

    def _get_model_params(self) -> ModelParamsType:
        """

        Returns
        -------
        ndarray
             Returns the prototypes as ndarray.

        """
        return self.prototypes_

    ###########################################################################################
    # Transformation (Params to variables and back) functions
    ###########################################################################################

    def _to_variables(self, model_params: ModelParamsType) -> np.ndarray:
        """

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        Returns
        -------
        ndarray
            Concatenated list of the model's parameters ravelled (ravel())

        """
        return model_params.ravel()

    def _to_params(self, variables: np.ndarray) -> ModelParamsType:
        """

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
        return np.reshape(variables, self.prototypes_.shape)

    ###########################################################################################
    # Other required functions (used in certain solvers)
    ###########################################################################################

    def _normalize_params(self, model_params: ModelParamsType) -> ModelParamsType:
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
        return LVQBaseClass._normalize_prototypes(model_params)

    ###########################################################################################
    # Initialization required functions
    ###########################################################################################

    def _initialize(self, data: np.ndarray, y: np.ndarray) -> SolverBaseClass:
        """
        Initialize is called by the LVQ base class and is required to do two things in order to
        work:
            1. It must initialize the distance functions and store it in 'self._distance'
            2. It must initialize the solver and return it.

        Besides these two things this is the function that should initialize any other algorithm
        specific parameters (besides the prototypes which are initialized in the base class)

        Parameters
        ----------
        data : ndarray with shape (number of observations, number of dimensions)
            Provided for models which require the data for initialization.
        y : ndarray with size equal to the number of observations
            Provided for models which require the labels for initialization.

        Returns
        -------
        solver
            Solver is either a subclass of SolverBaseClass or is an custom object that implements
            the required functions (see SolverBaseClass documentation).

        """
        distance_params = {"force_all_finite": self.force_all_finite}

        if self.distance_params is not None:
            distance_params.update(self.distance_params)

        self._distance = distances.grab(
            self.distance_type,
            class_kwargs=distance_params,
            whitelist=DISTANCE_FUNCTIONS,
        )

        # The objective is fixed as this determines what else to initialize.
        self._objective = GeneralizedLearningObjective(
            self.activation_type,
            self.activation_params,
            self.discriminant_type,
            self.discriminant_params,
        )

        solver = solvers.grab(
            self.solver_type,
            class_args=[self._objective],
            class_kwargs=self.solver_params,
            whitelist=SOLVERS,
        )

        return solver
