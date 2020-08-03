from . import LVQBaseClass

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.base import TransformerMixin

from sklvq import activations, discriminants, objectives, distances, solvers
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple, Union, List

from ..solvers import SolverBaseClass

ModelParamsType = Tuple[np.ndarray, np.ndarray]

DISTANCE_FUNCTIONS = [
    "local-adaptive-squared-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


# TODO: Could use different step-sizes for matrices
class LGMLVQ(LVQBaseClass):
    r"""Localized Generalized Matrix Learning Vector Quantization

    This model optimizes the generalized learning objective introduced by Sato and Yamada (
    1996). Additionally, it learns a relevance matrix (lambda\_ = omega\_.T.dot(omega\_)) in a local
    setting. This can either be per class or per prototype. The relevant omega per prototype is
    considered when computing the adaptive distance as introduced by Schneider et al. 2009.

    Parameters
    ----------
    distance_type : "local-adaptive-squared-euclidean" or Class
        Distance function that employs multiple relevance matrix in its calculation. This is
        controlled by the localization setting.

    distance_params : Dict, default=None
        Parameters passed to init of distance callable

    activation_type : {"identity", "sigmoid", "soft+", "swish"} or Class, default="sigmoid"
        The activation function used in the objective function.

    activation_params : Dict, default=None
        Parameters passed to init of activation function. See the documentation of activation
        functions for function dependent parameters and defaults.

    discriminant_type : "relative-distance" or Class
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

    initial_prototypes : "class-conditional-mean" or np.ndarray, default="class-conditional-mean"
        Default will initiate the prototypes to the class conditional mean with a small random
        offset. Custom numpy array can be passed to change the initial positions of the prototypes.

    prototypes_per_class : int or np.ndarray of length=n_prototypes, default=1
        Number of prototypes per class. Default will generate single prototype per class. In the
        case of unequal number of prototypes per class is preferable provide the labels as
        np.ndarray. Example prototypes_per_class = np.array([0, 0, 1, 2, 2, 2]) this will match
        with a total of 6 prototypes with first two class with index 0, then one with class index 1,
        and three with class index 2. Note: labels are indexes to classes\_ attribute.

    initial_omega : "identity" or np.ndarray, default="identity"
        Default will initiate the omega matrices to be the identity matrix. Other behaviour can
        be implemented by providing a custom omega as numpy array. E.g. a randomly initialized
        square matrix (n_features, n_features). The rank of the matrix can be reduced by
        providing a square matrix of shape ([1, n_features), n_features) (Bunte et al. 2012).

    localization : {"prototype", "class"}, default="prototype"
        Setting that controls the localization of the relevance matrices. Either per prototypes,
        where each prototype has its own relevance matrix. Or per class where each class has its
        own relevance matrix and if more then a single prototype per class is used it would be
        shared between these prototypes.

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
        Omega\_ matrices that were found during training and define the relevance matrices lambda\_.

    lambda_: np.ndarray of size (n_features, n_features)
        The relevance matrices (omega\_.T.dot(omega\_) per omega\_)

    omega_hat_: np.ndarray
        The omega matrices found by the eigenvalue decomposition of the relevance matrices lambda\_.
        The eigenvectors (columns of omega_hat\_) can be used to transform the data (Bunte et al.
        2012). This results in multiple possible transformations one per relevance matrix.

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
        distance_type: Union[str, type] = "local-adaptive-squared-euclidean",
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
        initial_omega_shape: Union[str, Tuple[int, int]] = "square",
        localization: str = "prototype",
        normalized_omega: bool = True,
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, int] = True,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.initial_omega = initial_omega
        self.initial_omega_shape = initial_omega_shape
        self.normalized_omega = normalized_omega
        self.localization = localization

        super(LGMLVQ, self).__init__(
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

    def set_model_params(self, model_params: ModelParamsType):
        """ Changes the model's internal parameters.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        (self.prototypes_, omega) = model_params

        if self.normalized_omega:
            self.omega_ = LGMLVQ._normalize_omega(omega)
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
            LGMLVQ._normalize_omega(omega),
        )

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
    def _normalize_omega(omega: np.ndarray) -> np.ndarray:
        denominator = np.sqrt(np.einsum("ikj, ikj -> i", omega, omega)).reshape(
            omega.shape[0], 1, 1
        )
        return omega / denominator

    def _initialize_omega(self, data: np.ndarray):
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
            self.omega_ = LGMLVQ._normalize_omega(self.omega_)

    ###########################################################################################
    # Transformer related functions
    ###########################################################################################

    @staticmethod
    def _compute_lambdas(omegas):
        # Equivalent to omega.T.dot(omega) per omega
        return np.einsum("ikj, ikl -> ijl", omegas, omegas)

    def _after_fit(self, data: np.ndarray, y: np.ndarray):
        self.lambda_ = LGMLVQ._compute_lambdas(self.omega_)
        # self.lambda_ = np.einsum("ikj, ikl -> ijl", self.omega_, self.omega_)

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

    def fit_transform(
        self, data: np.ndarray, y: np.ndarray, **trans_params
    ) -> np.ndarray:
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
        The data projected on columns of each omega\_hat\_ with shape (n_omegas, n_samples,
        n_columns)

        """
        return self.fit(data, y).transform(data, **trans_params)

    def transform(
        self,
        data: np.ndarray,
        scale: bool = False,
        omega_hat_index: Union[int, List[int]] = 0,
    ) -> np.ndarray:
        r"""

        Parameters
        ----------
        data : np.ndarray with shape (n_samples, n_features)
            Data that needs to be transformed
        scale : {True, False}, default = False
            Controls if the eigenvectors the data is projected on are scaled by the square root
            of their eigenvalues.
        omega_hat_index : int or list
            The indices of the omega\_hats\_ the transformation should be computed for.

        Returns
        -------
        The data projected on columns of each omega\_hat\_ with shape (n_omegas, n_samples,
        n_columns)

        """
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

    def _more_tags(self):
        # For some reason lgmlvq does not perform well on one of the test cases build into
        # sklearn's test cases.
        return {"poor_score": True}
