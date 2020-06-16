from sklvq.distances import DistanceBaseClass
from sklvq.models import LVQBaseClass

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import TransformerMixin

from sklvq import activations, discriminants, objectives, distances, solvers
from sklvq.objectives import GeneralizedLearningObjective

from typing import Union
from typing import Tuple

from sklvq.solvers import SolverBaseClass

ModelParamsType = tuple(np.ndarray, np.ndarray)

# TODO: Transform (inverse_transform) function sklearn

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
    "adaptive-squared-euclidean",
]

NAN_DISTANCE_FUNCTIONS = [
    "adaptive-squared-nan-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


class GMLVQ(LVQBaseClass, TransformerMixin):
    omega_: np.ndarray
    lambda_: np.ndarray
    omega_hat_: np.ndarray
    eigenvalues_: np.ndarray

    def __init__(
        self,
        distance_type="adaptive-squared-euclidean",
        distance_params=None,
        activation_type="identity",
        activation_params=None,
        discriminant_type="relative-distance",
        discriminant_params=None,
        solver_type="steepest-gradient-descent",
        solver_params=None,
        verbose=False,
        initial_prototypes="class-conditional-mean",
        prototypes_per_class=1,
        initial_omega="identity",
        normalized_omega=True,
        random_state=None,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.initial_omega = initial_omega
        self.normalized_omega = normalized_omega
        self.verbose = verbose

        super(GMLVQ, self).__init__(
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
            self.omega_ = GMLVQ.normalise_omega(omega)
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
        return (
            np.reshape(variables[0 : self.prototypes_.size], self.prototypes_.shape),
            np.reshape(variables[self.prototypes_.size :], self.omega_.shape),
        )

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
        (prototypes, omega) = model_params
        normalized_prototypes = LVQBaseClass.normalize_prototypes(prototypes)
        normalized_omega = GMLVQ.normalise_omega(omega)
        return normalized_prototypes, normalized_omega

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
    def normalise_omega(omega: np.ndarray) -> np.ndarray:
        norm_omega = omega / np.sqrt(np.einsum("ji, ji", omega, omega))
        return norm_omega

    def initialize_omega(self, data):
        if isinstance(self.initial_omega, np.ndarray):
            # TODO Checks
            self.omega_ = self.initial_omega
        elif self.initial_omega == "identity":
            self.omega_ = np.eye(data.shape[1])
        else:
            raise ValueError("The provided value for the parameter 'omega' is invalid.")

        if self.normalized_omega:
            self.omega_ = GMLVQ.normalise_omega(self.omega_)

    ###########################################################################################
    # Transformer related functions
    ###########################################################################################

    def after_fit(self, data: np.ndarray, y: np.ndarray):
        self.lambda_ = self.omega_.T.dot(self.omega_)

        eigenvalues, omega_hat = np.linalg.eig(self.lambda_)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.omega_hat_ = omega_hat[:, sorted_indices]

    def fit_transform(self, data: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(data, y).transform(data)

    def transform(self, data: np.ndarray, scale: bool = False) -> np.ndarray:
        data = check_array(data)

        check_is_fitted(self)

        transformation_matrix = self.omega_hat_
        if scale:
            transformation_matrix = np.sqrt(self.eigenvalues_) * transformation_matrix

        data_new = data.dot(transformation_matrix)

        return data_new

    # TODO: add a sklvq.plot for these things?
    # def dist_function(self, data):
    #     # SciKit-learn list of checked params before predict
    #     check_is_fitted(self)
    #
    #     # Input validation
    #     data = check_array(data)
    #
    #     distances = self.distance_(data, self)
    #     min_args = np.argsort(distances, axis=1)
    #
    #     winner = distances[list(range(0, distances.shape[0])), min_args[:, 0]]
    #     runner_up = distances[list(range(0, distances.shape[0])), min_args[:, 1]]
    #
    #     return np.abs(winner - runner_up) / (
    #         2
    #         * np.linalg.norm(
    #             self.prototypes_[min_args[:, 0], :]
    #             - self.prototypes_[min_args[:, 1], :]
    #         )
    #         ** 2
    #     )

    # def rel_dist_function(self, data):
    #     # SciKit-learn list of checked params before predict
    #     check_is_fitted(self)
    #
    #     # Input validation
    #     data = check_array(data)
    #
    #     distances = np.sort(self.distance_(data, self))
    #
    #     winner = distances[:, 0]
    #     runner_up = distances[:, 1]
    #
    #     return (runner_up - winner) / (winner + runner_up)
    #
    # def d_plus_function(self, data):
    #     # SciKit-learn list of checked params before predict
    #     check_is_fitted(self)
    #
    #     # Input validation
    #     data = check_array(data)
    #
    #     distances = np.sort(self.distance_(data, self))
    #
    #     winner = distances[:, 0]
    #
    #     return -1 * winner
