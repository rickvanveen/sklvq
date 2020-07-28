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

ModelParamsType = Tuple[np.ndarray, np.ndarray]

# TODO: Transform (inverse_transform) function sklearn

DISTANCE_FUNCTIONS = [
    "adaptive-squared-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


class GMLVQ(LVQBaseClass):
    r"""Generalized Matrix Learning Vector Quantization

    This model optimizes the generalized learning objective introduced by Sato and Yamada (
    1996). Additionally, it learns a relevance matrix that is used in the distance functions as
    introduced by Schneider et al. (2009)

    Parameters
    ----------
    distance_type : {"adaptive-squared-euclidean"} or Class, default="squared-euclidean"
        Distance function that employs a relevance matrix in its calculation.

    distance_params : Dict, default=None
        Parameters passed to init of distance callable

    activation_type : {"identity", "sigmoid", "soft+", "swish"} or Class, default="sigmoid"
        The activation function used in the objective function.

    activation_params : Dict, default=None
        Parameters passed to init of activation function. See the documentation of activation
        functions for function dependent parameters and defaults.

    discriminant_type : {"relative-distance"} or Class, default = "relative-distance"
        The discriminant function.

    discriminant_params : Dict, default=None
        Parameters passed to init of discriminant callable

    solver_type : {"sgd", "wgd", "adam", "lbfgs", "bfgs"},
        The solver used for optimization

        - "sgd" is an alias for the steepest gradient descent solver. Implements both the
            stochastic  and (mini) batch variants. Depending on chosen batch size.

        - "wgd" or waypoint gradient descent optimization (Papari et al. 2011)

        - "adam" also known as adaptive moment estimation. Implementation based on description
            by Lekander et al (2017)

        - "bfgs" or the broyden-fletcher-goldfarb-shanno optimization algorithm. Uses the scipy
            implementation.

        - "lbfgs" is an alias for limited-memory-bfgs with bfgs the same as above. Uses the
            scipy implementation.

    solver_params : Dict, default=None
        Parameters passed to init of solvers. See the documentation of the solver
        functions for relevant parameters and defaults.

    initial_prototypes : {"class-conditional-mean"} or np.ndarray, default="class-conditional-mean"
        Default will initiate the prototypes to the class conditional mean with a small random
        offset. Custom numpy array can be passed to change the initial positions of the prototypes.

    prototypes_per_class : int or np.ndarray of length=n_prototypes, default=1
        Number of prototypes per class. Default will generate single prototype per class. In the
        case of unequal number of prototypes per class is preferable provide the labels as
        np.ndarray. Example prototypes_per_class = np.array([0, 0, 1, 2, 2, 2]) this will match
        with a total of 6 prototypes with first two class with index 0, then one with class index 1,
        and three with class index 2. Note: labels are indexes to classes\_ attribute.

    initial_omega : {"identity"} or np.ndarray, default="identity"
        Default will initiate the omega matrices to be the identity matrix. Other behaviour can
        be implemented by providing a custom omega as numpy array. E.g. a randomly initialized
        square matrix (n_features, n_features). The rank of the matrix can be reduced by
        providing a square matrix of shape ([1, n_features), n_features) (Bunte et al. 2012).

    normalized_omega : {True, False}, default=True
        Flag to indicate whether to normalize omega such that the trace of the relevance matrix
        is (approximately) equal to 1.

    random_state : int, RandomState instance, default=None
        Determines random number generation.

    force_all_finite : {True, "allow-nan"}, default=True
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:

        - True: Force all values of array to be finite.
        - "allow-nan": accepts only np.nan and pd.NA values in array. Values cannot be infinite.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Class labels for each output.

    prototypes_ : np.ndarray of shape (n_protoypes, n_features)
        Positions of the prototypes after fit(data, labels) has been called.

    prototypes_labels_ : np.ndarray of shape (n_prototypes)
        Labels for each prototypes. Labels are indexes to classes\_

    omega_: np.ndarray with size depending on initialization, default (n_features, n_features)
        Omega matrix that was found during training and defines the relevance matrix lambda\_.

    lambda_: np.ndarray of size (n_features, n_features)
        The relevance matrix = omega\_.T.dot(omega\_)

    omega_hat_: np.ndarray
        The omega matrix found by the eigenvalue decomposition of the relevance matrix lambda\_.
        The eigenvectors (columns of omega_hat\_) can be used to transform the data (Bunte et al.
        2012).

    eigenvalues_: np.ndarray
        The corresponding eigenvalues to omega_hat\_ found by the eigenvalue decomposition of
        the relevance matrix lambda\_

    References
    ----------
    Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
    Advances in Neural Network Information Processing Systems, 423–429, 1996.

    Schneider, P., Biehl, M., & Hammer, B. (2009). "Adaptive Relevance Matrices in Learning Vector
    Quantization" Neural Computation, 21(12), 3532–3561, 2009.

    Papari, G., and Bunte, K., and Biehl, M. (2011) "Waypoint averaging and step size control in
    learning by gradient descent" Mittweida Workshop on Computational Intelligence (MIWOCI) 2011.

    LeKander, M., Biehl, M., & De Vries, H. (2017). "Empirical evaluation of gradient methods for
    matrix learning vector quantization." 12th International Workshop on Self-Organizing Maps and
    Learning Vector Quantization, Clustering and Data Visualization, WSOM 2017.

    Bunte, K., Schneider, P., Hammer, B., Schleif, F.-M., Villmann, T., & Biehl, M. (2012).
    "Limited Rank Matrix Learning, discriminative dimension reduction and visualization." Neural
    Networks, 26, 159–173, 2012.
"""

    classes_: np.ndarray
    prototypes_: np.ndarray
    prototypes_labels_: np.ndarray
    omega_: np.ndarray
    lambda_: np.ndarray
    omega_hat_: np.ndarray
    eigenvalues_: np.ndarray

    def __init__(
        self,
        distance_type: Union[str, type] = "adaptive-squared-euclidean",
        distance_params: dict = None,
        activation_type: Union[str, type] = "identity",
        activation_params: dict = None,
        discriminant_type: Union[str, type] = "relative-distance",
        discriminant_params: dict = None,
        solver_type: Union[str, type] = "steepest-gradient-descent",
        solver_params: dict = None,
        initial_prototypes: Union[str, np.ndarray] = "class-conditional-mean",
        prototypes_per_class: Union[int, np.ndarray] = 1,
        initial_omega: Union[str, np.ndarray] = "identity",
        normalized_omega: bool = True,
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, bool] = True,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.initial_omega = initial_omega
        self.normalized_omega = normalized_omega

        super(GMLVQ, self).__init__(
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

    def _set_model_params(self, model_params: ModelParamsType):
        """ Changes the model's internal parameters.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        (self.prototypes_, omega) = model_params

        if self.normalized_omega:
            self.omega_ = GMLVQ._normalise_omega(omega)
        else:
            self.omega_ = omega

    def _get_model_params(self) -> ModelParamsType:
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
        omega_size = self.omega_.size
        prototypes_size = self.prototypes_.size

        variables = np.zeros(prototypes_size + omega_size)

        (variables[0:prototypes_size], variables[prototypes_size:]) = map(
            np.ravel, model_params
        )

        return variables

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
        return (
            np.reshape(variables[0 : self.prototypes_.size], self.prototypes_.shape),
            np.reshape(variables[self.prototypes_.size :], self.omega_.shape),
        )

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
        (prototypes, omega) = model_params
        normalized_prototypes = LVQBaseClass._normalize_prototypes(prototypes)
        normalized_omega = GMLVQ._normalise_omega(omega)
        return normalized_prototypes, normalized_omega

    ###########################################################################################
    # Initialization functions
    ###########################################################################################

    def _initialize(self, data: np.ndarray, y: np.ndarray) -> SolverBaseClass:
        """ Initialize is called by the LVQ base class and is required to do two things in order to
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
        self._initialize_omega(data)

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

    ###########################################################################################
    # Algorithm specific functions
    ###########################################################################################

    @staticmethod
    def _normalise_omega(omega: np.ndarray) -> np.ndarray:
        norm_omega = omega / np.sqrt(np.einsum("ji, ji", omega, omega))
        return norm_omega

    def _initialize_omega(self, data):
        if isinstance(self.initial_omega, np.ndarray):
            # TODO Checks
            self.omega_ = self.initial_omega
        elif self.initial_omega == "identity":
            self.omega_ = np.eye(data.shape[1])
        else:
            raise ValueError("The provided value for the parameter 'omega' is invalid.")

        if self.normalized_omega:
            self.omega_ = GMLVQ._normalise_omega(self.omega_)

    @staticmethod
    def _compute_lambda(omega):
        # Equivalent to omega.T.dot(omega)
        return np.einsum("ji, jk -> ik", omega, omega)

    ###########################################################################################
    # Transformer related functions
    ###########################################################################################

    def _after_fit(self, data: np.ndarray, y: np.ndarray):
        # self.lambda_ = self.omega_.T.dot(self.omega_)
        self.lambda_ = GMLVQ._compute_lambda(self.omega_)

        eigenvalues, omega_hat = np.linalg.eig(self.lambda_)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.omega_hat_ = omega_hat[:, sorted_indices]

    def fit_transform(self, data: np.ndarray, y: np.ndarray, **trans_params) -> np.ndarray:
        r"""

        Parameters
        ----------
        data : np.ndarray with shape (n_samples, n_features)
            Data used for fit and that will be transformed.
        y : np.ndarray with length (n_samples)
            Labels corresponding to the data samples.
        trans_params :
            Parameters passed to transform function

        Returns
        -------
        The data projected on columns of omega\_hat\_ with shape (n_samples, n_columns)

        """
        return self.fit(data, y).transform(data, **trans_params)

    def transform(self, data: np.ndarray, scale: bool = False) -> np.ndarray:
        r"""

        Parameters
        ----------
        data : np.ndarray with shape (n_samples, n_features)
            Data that needs to be transformed
        scale : {True, False}, default = False
            Controls if the eigenvectors the data is projected on are scaled by the square root
            of their eigenvalues.

        Returns
        -------
        The data projected on columns of omega\_hat\_ with shape (n_samples, n_columns)

        """
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
    #     distances = self._distance(data, self)
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
    #     distances = np.sort(self._distance(data, self))
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
    #     distances = np.sort(self._distance(data, self))
    #
    #     winner = distances[:, 0]
    #
    #     return -1 * winner
