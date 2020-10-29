from typing import Tuple
from typing import Union

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from . import LVQBaseClass
from ..objectives import GeneralizedLearningObjective

ModelParamsType = Tuple[np.ndarray, np.ndarray]

DISTANCES = [
    "adaptive-squared-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]

_RELEVANCES_PARAMS_DEFAULS = {
    "normalization": True,
    "n_components": "all",
}


class GMLVQ(LVQBaseClass):
    r"""Generalized Matrix Learning Vector Quantization

    This model uses the :class:`sklvq.objectives.GeneralizedLearningObjective` as its objective
    function [1]_. In addition to learning the positions of the prototypes it learns a relevance
    matrix that is used in the distance functions [2]_.

    Parameters
    ----------
    distance_type : {"adaptive-squared-euclidean"} or Class, default="squared-euclidean"
        Distance function that employs a relevance matrix in its calculation.

        - "adaptive-squared-euclidean"
            See :class:`sklvq.distances.AdaptiveSquaredEuclidean`

    distance_params : Dict, default=None
        Parameters passed to init of distance callable

    activation_type : {"identity", "sigmoid", "soft+", "swish"} or Class, default="sigmoid"
        Parameters passed to init of activation function. See the documentation of the activation
        functions for parameters and defaults.

        - "identity"
            See :class:`sklvq.activations.Identity`
        - "sigmoid"
            See :class:`sklvq.activations.Sigmoid`
        - "soft+"
            See :class:`sklvq.activations.SoftPlus`
        - "swish"
            See :class:`sklvq.activations.Swish`

    activation_params : Dict, default=None
        Parameters passed to init of activation function. See the documentation of activation
        functions for function dependent parameters and defaults.

    discriminant_type : {"relative-distance"} or Class, default = "relative-distance"
        The discriminant function.  Note that different discriminant type may require to rewrite
        the ``decision_function`` and ``predict_proba`` methods.

        - "relative-distance"
            See :class:`sklvq.discriminants.RelativeDistance`

    discriminant_params : Dict, default=None
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
            Implementation from scipy package.
        - "lbfgs" or "limited-memory-bfgs"
            Implementation from scipy package.

    solver_params : dict, default=None
        Parameters passed to init of solvers. See the documentation of the solvers relevant
        parameters and defaults.

    prototype_init: "class-conditional-mean" or ndarray, default="class-conditional-mean"
        Default will initiate the prototypes to the class conditional mean with a small random
        offset. Custom numpy array can be passed to change the initial positions of the prototypes.

    prototype_params: dict = None,
        Containing the following parameters (keys):

        - "protoytpes_per_class":  int or ndarray, optional, default=1
            Default will generate single prototype per class. In the case of unequal number of
            prototypes per class is needed, provide the labels as  np.ndarray. For example,
            prototypes_per_class = np.array([0, 0, 1, 2, 2, 2]) this will result in a  total of 6
            prototypes with the first two classes with index 0, then one with class index 1,
            and three with class index 2. Note: labels are indexes to classes\_ attribute, which is
            equal to np.unique(labels)

    relevance_init : {"identity", "random"} or np.ndarray, default="identity"
        Default will initiate the omega matrices to be the identity matrix. Other behaviour can
        be implemented by providing a custom omega as numpy array. E.g. a randomly initialized
        square matrix (n_features, n_features). The rank of the matrix can be reduced by
        providing a square matrix of shape ([1, n_features), n_features)  [3]_.

    relevance_params: dict = None,
        Containing the following parameters (keys):

        - "normalized_omega" : {True, False}, default=True
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
    classes_ : ndarray of shape (n_classes,)
        Class labels for each output.

    prototypes_ : ndarray of shape (n_protoypes, n_features)
        Positions of the prototypes after ``fit(X, labels)`` has been called.

    prototypes_labels_ : ndarray of shape (n_prototypes)
        Labels for each prototypes. Labels are indexes to ``classes_``

    omega_: ndarray with size depending on initialization, default (n_features, n_features)
        Omega matrix that was found during training and defines the relevance matrix ``lambda_``.

    lambda_: ndarray of size (n_features, n_features)
        The relevance matrix ``omega_.T.dot(omega_)``

    omega_hat_: ndarray
        The omega matrix found by the eigenvalue decomposition of the relevance matrix ``lambda_``.
        The eigenvectors (columns of ``omega_hat_``) can be used to transform the X [3]_.

    eigenvalues_: ndarray
        The corresponding eigenvalues to ``omega_hat_`` found by the eigenvalue decomposition of
        the relevance matrix ``lambda_``

    References
    ----------
    .. [1] Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
        Advances in Neural Network Information Processing Systems, 423–429, 1996.

    .. [2] Schneider, P., Biehl, M., & Hammer, B. (2009). "Adaptive Relevance Matrices in
        Learning Vector Quantization" Neural Computation, 21(12), 3532–3561, 2009.

    .. [3] Bunte, K., Schneider, P., Hammer, B., Schleif, F.-M., Villmann, T., & Biehl, M. (2012).
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
        activation_type: Union[str, type] = "sigmoid",
        activation_params: dict = None,
        discriminant_type: Union[str, type] = "relative-distance",
        discriminant_params: dict = None,
        solver_type: Union[str, type] = "steepest-gradient-descent",
        solver_params: dict = None,
        prototype_init: Union[str, np.ndarray] = "class-conditional-mean",
        prototype_params: dict = None,
        relevance_init="identity",
        relevance_params: dict = None,
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, bool] = True,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.relevance_init = relevance_init
        relevance_params = self._init_model_params_options(
            relevance_params, _RELEVANCES_PARAMS_DEFAULS
        )
        self.relevance_params = relevance_params

        super(GMLVQ, self).__init__(
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

    def set_variables(self, new_variables: np.ndarray) -> None:
        """

        Modifies the ``self._variables`` by copying the values of ``new_variables`` into the
        memory of ``self._variables``.

        Parameters
        ----------
        new_variables : ndarray
            1d numpy array that contains all the model parameters in continuous memory

        """
        np.copyto(self._variables, new_variables)
        if self._needs_normalizing():
            GMLVQ._normalise_omega(self.omega_)

    def set_model_params(self, new_model_params: ModelParamsType):
        """
        Changes the model's internal parameters. Copies the values of model_params into
        ``self.prototypes_`` and ``self.omega_`` therefor updating the ``self.variables_``
        array.

        Also normalized the relevance matrix if necessary.

        Parameters
        ----------
        new_model_params : tuple of ndarrays
            Shapes depend on  initialization but in the case of a square relevance matrix:
            tuple((n_prototypes, n_features), (n_features, n_features))

        """
        new_prototypes, new_omega = new_model_params

        self.set_prototypes(new_prototypes)
        self.set_omega(new_omega)

        if self._needs_normalizing():
            GMLVQ._normalise_omega()

    def get_model_params(self) -> ModelParamsType:
        """
        Returns a tuple of all model parameters. In this case the prototypes and omega matrix.

        Returns
        -------
        ndarray
             Returns a tuple of views, i.e., the prototypes and omega matrix.

        """
        return self.prototypes_, self.omega_

    ###########################################################################################
    # Specific "getters" and "setters" for GMLVQ
    ###########################################################################################

    def get_omega(self):
        """
        Convenience function to return ``self.omega_``

        Returns
        -------
            ndarray, with shape depending on initialization of omega.

        """
        return self.omega_

    def set_omega(self, omega):
        """
        Convenience function that makes sure to copy the value to ``self.omega_`` and not overwrite
        it.

        Parameters
        ----------
        omega : ndarray with same shape as ``self.omega_``

        """
        np.copyto(self.omega_, omega)

    ###########################################################################################
    # Functions to transform the 1D variables array to model parameters and back
    ###########################################################################################

    def to_model_params_view(self, var_buffer: np.ndarray) -> ModelParamsType:
        """

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        tuple
            Returns a tuple with the prototypes and omega matrix as ndarrays.

        """
        return (
            self.to_prototypes_view(var_buffer),
            self.to_omega(var_buffer),
        )

    def to_prototypes_view(self, var_buffer: np.ndarray) -> np.ndarray:
        """
        Returns a view (of the shape of the model's prototypes) into the provided variables
        buffer of the same size as the model's variables array.

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
        return var_buffer[: self._prototypes_size].reshape(self._prototypes_shape)

    def to_omega(self, var_buffer: np.ndarray) -> np.ndarray:
        """
        Returns a view (of the shape of the model's omega) into the provided variables
        buffer of the same size as the model's variables array.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        ndarray
            Shape depending on initialization but in case of a square matrix (n_features,
            n_features.

        """
        return var_buffer[self._prototypes_size :].reshape(self._relevances_shape)

    ###########################################################################################
    # Solver Normalization functions
    ###########################################################################################

    def normalize_variables(self, var_buffer: np.ndarray) -> None:
        """
        Modifies the var_buffer as if it was the variables array provided
        by ``get_variables()``. Will select, reshape and normalize the correct parts of the
        variable buffer.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        """
        (prototypes, omega) = self.to_model_params_view(var_buffer)

        self._normalize_prototypes(prototypes)
        self._normalise_omega(omega)

    @staticmethod
    def _normalise_omega(omega: np.ndarray) -> None:
        np.divide(omega, np.sqrt(np.einsum("ji, ji", omega, omega)), out=omega)

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

        omega_view = gradient[self.prototypes_.size :]
        np.add(omega_view, partial_gradient[n_features:], out=omega_view)

    def mul_step_size(
        self, step_sizes: Union[int, float, np.ndarray], gradient: np.ndarray
    ) -> None:
        """
        If step sizes is a scalar value just multiplies the gradient with the step size. If it
        is an array (with same length as number of model parameters) each model parameter is
        multiplied by its own step size.

        Parameters
        ----------
        step_sizes : float or ndarray
            The scalar or list of values containing the step sizes.
        gradient : ndarray
            Same shape as the ``get_variables()`` would return.

        """
        if isinstance(step_sizes, int) | isinstance(step_sizes, float):
            gradient *= step_sizes
            return

        if isinstance(step_sizes, np.ndarray):
            if step_sizes.size > 1:
                prototypes, omega = self.to_model_params_view(gradient)
                prototypes *= step_sizes[0]
                omega *= step_sizes[1]
            # if size is more than 2 it just ignores the additional values.

    ###########################################################################################
    # Initialization functions
    ###########################################################################################

    def _init_variables(self) -> None:
        self._variables = np.empty(
            self._prototypes_size + self._relevances_size, dtype="float64", order="C",
        )

    def _check_model_params(self):
        self._check_prototype_params(**self.prototype_params)
        self._check_relevances_params(**self.relevance_params)

    def _init_model_params(self, X, y) -> None:
        self._init_prototypes(X, y, **self.prototype_params)
        self._init_relevances(**self.relevance_params)

    def _check_relevances_params(
        self, normalization=True, n_components: Union[str, int] = "all"
    ):
        if not isinstance(normalization, bool):
            raise ValueError("Provided normalization is invalid.")

        if n_components == "all":
            self._relevances_shape = (self.n_features_in_, self.n_features_in_)
        elif n_components >= 1 and n_components <= self.n_features_in_:
            self._relevances_shape = (self.n_features_in_, n_components)
        else:
            raise ValueError("Provided n_components is invalid.")

        self._relevances_size = np.prod(self._relevances_shape)

    def _init_relevances(self, normalization=True, n_components="all"):
        self.omega_ = self.to_omega(self._variables)

        if isinstance(self.relevance_init, str):
            if self.relevance_init == "identity":
                self.set_omega(np.eye(*self._relevances_shape))
            elif self.relevance_init == "random":
                self.set_omega(
                    self.random_state_.uniform(
                        low=0, high=1, size=self._relevances_shape
                    )
                )
            else:
                raise ValueError("Provided relevance_init is invalid.")
        else:
            raise ValueError("Provided relevance_init is invalid.")

        if normalization:
            GMLVQ._normalise_omega(self.omega_)

    def _init_objective(self):
        self._objective = GeneralizedLearningObjective(
            self.activation_type,
            self.activation_params,
            self.discriminant_type,
            self.discriminant_params,
        )

    ###########################################################################################
    # Other Algorithm specific stuff.
    ###########################################################################################

    def _after_fit(self, X: np.ndarray, y: np.ndarray):
        # self.lambda_ = self.omega_.T.dot(self.omega_)
        self.lambda_ = GMLVQ._compute_lambda(self.omega_)

        eigenvalues, omega_hat = np.linalg.eig(self.lambda_)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.omega_hat_ = omega_hat[:, sorted_indices]

    @staticmethod
    def _compute_lambda(omega):
        # Equivalent to omega.T.dot(omega), but faster (?)
        return np.einsum("ji, jk -> ik", omega, omega)

    def _needs_normalizing(self):
        return self.relevance_params.get("normalization", True)

    ###########################################################################################
    # Transformer related functions
    ###########################################################################################

    def fit_transform(
        self, data: np.ndarray, y: np.ndarray, **transform_params
    ) -> np.ndarray:
        r"""

        Parameters
        ----------
        data : ndarray with shape (n_samples, n_features)
            Data used for fit and that will be transformed.
        y : np.ndarray with length (n_samples)
            Labels corresponding to the X samples.
        trans_params :
            Parameters passed to transform function

        Returns
        -------
        The data projected on columns of ``omega_hat_`` with shape (n_samples, n_columns)

        """
        return self.fit(data, y).transform(data, **transform_params)

    def transform(self, X: np.ndarray, scale: bool = False) -> np.ndarray:
        r"""

        Parameters
        ----------
        X : np.ndarray with shape (n_samples, n_features)
            Data that needs to be transformed
        scale : {True, False}, default = False
            Controls if the eigenvectors the data is projected on are scaled by the square root
            of their eigenvalues.

        Returns
        -------
        The data projected on columns of ``omega_hat_`` with shape (n_samples, n_columns)

        """
        X = check_array(X)

        check_is_fitted(self)

        transformation_matrix = self.omega_hat_
        if scale:
            transformation_matrix = np.sqrt(self.eigenvalues_) * transformation_matrix

        data_new = X.dot(transformation_matrix)

        return data_new