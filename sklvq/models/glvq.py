from . import LVQBaseClass
import numpy as np

# Can be switched out by parameters to the models.
from sklvq import activations, discriminants, distances, solvers

# Typing
from sklvq.solvers import SolverBaseClass
from sklvq.distances import DistanceBaseClass

# Cannot be switched out by parameters to the models.
from sklvq.objectives import GeneralizedLearningObjective


ModelParamsType = np.ndarray

DISTANCE_FUNCTIONS = [
    "euclidean",
    "squared-euclidean",
]

NAN_DISTANCE_FUNCTIONS = [
    "nan-euclidean",
    "squared-nan-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


class GLVQ(LVQBaseClass):
    def __init__(
        self,
        distance_type="squared-euclidean",
        distance_params=None,
        activation_type="sigmoid",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
        solver_type="steepest-gradient-descent",
        solver_params=None,
        initial_prototypes="class-conditional-mean",
        prototypes_per_class=1,
        random_state=None,
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
            prototypes_per_class,
            initial_prototypes,
            random_state,
        )

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    def set_model_params(self, model_params: ModelParamsType) -> None:
        """
        Changes the model's internal parameters.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        self.prototypes_ = model_params

    def get_model_params(self) -> ModelParamsType:
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

    def to_variables(self, model_params: ModelParamsType) -> np.ndarray:
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

    def to_params(self, variables: np.ndarray) -> ModelParamsType:
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

    def normalize_params(self, model_params: ModelParamsType) -> ModelParamsType:
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
        return LVQBaseClass.normalize_prototypes(model_params)

    ###########################################################################################
    # Initialization required functions
    ###########################################################################################

    def initialize(self, data: np.ndarray, y: np.ndarray) -> SolverBaseClass:
        """
        Initialize is called by the LVQ base class and is required to do two things in order to
        work:
            1. It must initialize the distance functions and store it in 'self.distance_'
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
        self.distance_ = distances.grab(
            self.distance_type,
            class_kwargs=self.distance_params,
            whitelist=DISTANCE_FUNCTIONS + NAN_DISTANCE_FUNCTIONS,
        )

        # The objective is fixed as this determines what else to initialize.
        self.objective_ = GeneralizedLearningObjective(
            self.activation_type,
            self.activation_params,
            self.discriminant_type,
            self.discriminant_params,
        )

        solver = solvers.grab(
            self.solver_type,
            class_args=[self.objective_],
            class_kwargs=self.solver_params,
            whitelist=SOLVERS,
        )

        return solver
