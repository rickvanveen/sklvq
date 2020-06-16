from . import LVQBaseClass

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import TransformerMixin

from sklvq import activations, discriminants, objectives, distances, solvers
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple, Union, List

from ..solvers import SolverBaseClass

ModelParamsType = tuple(np.ndarray, np.ndarray)

ACTIVATION_FUNCTIONS = [
    "identity",
    "sigmoid",
    "soft-plus",
    "swish",
]

DISCRIMINANT_FUNCTIONS = [
    "relative-distance",
]

DISTANCE_FUNCTIONS = [
    "local-adaptive-squared-euclidean",
]

NAN_DISTANCE_FUNCTIONS = [
    "local-adaptive-squared-nan-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


# TODO: Could use different step-sizes for matrices
class LGMLVQ(LVQBaseClass, TransformerMixin):
    lambda_: np.ndarray
    omega_hat_: np.ndarray
    eigenvalues_: np.ndarray

    def __init__(
        self,
        distance_type="local-adaptive-squared-euclidean",
        distance_params=None,
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
        solver_type="steepest-gradient-descent",
        solver_params=None,
        localization="prototype",
        verbose=False,
        initial_prototypes="class-conditional-mean",
        prototypes_per_class=1,
        initial_omega="identity",
        initial_omega_shape="square",
        normalized_omega=True,
        random_state=None,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.initial_omega = initial_omega
        self.initial_omega_shape = initial_omega_shape
        self.normalized_omega = normalized_omega
        self.localization = localization
        self.verbose = verbose

        super(LGMLVQ, self).__init__(
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
        (self.prototypes_, omega) = model_params

        if self.normalized_omega:
            self.omega_ = LGMLVQ.normalize_omega(omega)
        else:
            self.omega_ = omega

    def get_model_params(self) -> ModelParamsType:
        """

        Returns
        -------
        ndarray
             Returns the prototypes as ndarray.

        """
        return self.prototypes_, self.omega_

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
        omega_size = self.omega_.size
        prototypes_size = self.prototypes_.size

        variables = np.zeros(prototypes_size + omega_size)

        (variables[0:prototypes_size], variables[prototypes_size:]) = map(
            np.ravel, model_params
        )

        return variables

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
        prototypes_size = self.prototypes_.size
        return (
            np.reshape(variables[0:prototypes_size], self.prototypes_.shape),
            np.reshape(variables[prototypes_size:], self.omega_.shape),
        )

    ###########################################################################################
    # Other functions (used in waypoint-gradient-descent solver only)
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
        (prototypes, omega) = model_params

        return (
            LVQBaseClass.normalize_prototypes(prototypes),
            LGMLVQ.normalise_omega(omega),
        )

    ###########################################################################################
    # Initialization functions
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
        self.initialize_omega(data)

        self.distance_ = distances.grab(
            self.distance_type,
            class_kwargs=self.distance_params,
            whitelist=DISTANCE_FUNCTIONS + NAN_DISTANCE_FUNCTIONS,
        )

        activation = activations.grab(
            self.activation_type,
            class_kwargs=self.activation_params,
            whitelist=ACTIVATION_FUNCTIONS,
        )

        discriminant = discriminants.grab(
            self.discriminant_type,
            class_kwargs=self.discriminant_params,
            whitelist=DISCRIMINANT_FUNCTIONS,
        )

        # The objective is fixed as this determines what else to initialize.
        self.objective_ = GeneralizedLearningObjective(
            activation=activation, discriminant=discriminant
        )

        solver = solvers.grab(
            self.solver_type,
            class_args=[self.objective_],
            class_kwargs=self.solver_params,
            whitelist=SOLVERS,
        )

        return solver

    ###########################################################################################
    # Algorithm specific functions
    ###########################################################################################

    @staticmethod
    def normalize_omega(omega: np.ndarray) -> np.ndarray:
        denominator = np.sqrt(np.einsum("ikj, ikj -> i", omega, omega)).reshape(
            omega.shape[0], 1, 1
        )
        return omega / denominator

    def initialize_omega(self, data):
        # Custom omega?
        if self.initial_omega == "identity":
            if self.localization == "prototype":
                num_omega = self.prototypes_.shape[0]
            elif self.localization == "class":
                num_omega = self.classes_.size
            else:
                raise ValueError(
                    "The provided value for the parameter 'localization' is invalid."
                )
            if self.initial_omega_shape == "square":
                shape = (data.shape[1], data.shape[1])
            else:
                shape = self.initial_omega_shape
            self.omega_ = np.array([np.eye(*shape) for _ in range(num_omega)])
        else:  # Custom omega?
            # TODO Checks (localization.... custom labeling to prototypes...)
            self.omega_ = self.initial_omega
            # raise ValueError("The provided value for the parameter 'omega' is invalid.")

        if self.normalized_omega:
            self.omega_ = LGMLVQ.normalize_omega(self.omega_)

    ###########################################################################################
    # Transformer related functions
    ###########################################################################################

    def after_fit(self, data: np.ndarray, y: np.ndarray):
        self.lambda_ = np.einsum("ikj, ikl -> ijl", self.omega_, self.omega_)

        eigenvalues, omega_hat = np.linalg.eig(self.lambda_)

        sorted_indices = np.flip(np.argsort(eigenvalues, axis=1), axis=1)
        eigenvalues, omega_hat = zip(
            *[
                (lk[ii], ek[:, ii])
                for (ii, lk, ek) in zip(sorted_indices, eigenvalues, omega_hat)
            ]
        )
        self.eigenvalues_ = np.array(eigenvalues)
        self.omega_hat_ = np.array(omega_hat)

    def fit_transform(self, data: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(data, y).transform(data)

    def transform(
        self, data: np.ndarray, scale: bool = False, omega_hat_index: Union[int, List[int]] = 0,
    ) -> np.ndarray:
        check_is_fitted(self)

        data = check_array(data)

        transformation_matrix = self.omega_hat_[omega_hat_index, :, :]
        if transformation_matrix.ndim != 3:
            transformation_matrix = np.expand_dims(transformation_matrix, 0)

        if scale:
            transformation_matrix = np.einsum(
                "ik, ijk -> ijk", np.sqrt(self.eigenvalues_), transformation_matrix
            )
        transformed_data = np.einsum("jk, ikl -> ijl", data, transformation_matrix)

        return np.squeeze(transformed_data)
