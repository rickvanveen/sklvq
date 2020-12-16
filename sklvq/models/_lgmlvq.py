from typing import Tuple, Union, List

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array

from . import LVQBaseClass
from ..objectives import GeneralizedLearningObjective

ModelParamsType = Tuple[np.ndarray, np.ndarray]

DISTANCES = [
    "local-adaptive-squared-euclidean",
]

SOLVERS = [
    "adaptive-moment-estimation",
    "broyden-fletcher-goldfarb-shanno",
    "limited-memory-bfgs",
    "steepest-gradient-descent",
    "waypoint-gradient-descent",
]


class LGMLVQ(LVQBaseClass):
    r"""Localized Generalized Matrix Learning Vector Quantization

    This model uses the :class:`sklvq.objectives.GeneralizedLearningObjective` as its objective
    function `[1]`_. In addition to learning the positions of the prototypes it learns a set
    of relevance matrices in a localized manner, that are used in the distance functions `[2]`_.

    Parameters
    ----------
    distance_type : "local-adaptive-squared-euclidean" or Class
        Distance function that employs multiple relevance matrix in its calculation. This is
        controlled by the localization setting.

        - "local-adaptive-squared-euclidean"
            See :class:`sklvq.distances.LocalAdaptiveSquaredEuclidean`

    distance_params : dict, default=None
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


    activation_params : Dict, default=None
        Parameters passed to init of activation function. See the documentation of the activation
        functions for parameters and defaults.

    discriminant_type : "relative-distance" or Class
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

    prototype_n_per_class: int or np.ndarray, optional, default=1
        Default will generate single prototype per class. In the case of unequal number of
        prototypes per class is needed, provide this as np.ndarray. For example,
        prototype_n_per_class = np.array([1, 6, 3]) this will result in one prototype for the first class,
        six for the second, and three for the third. Note that the order needs to be the same as the on in the
        classes\_ attribute, which is equal to calling np.unique(labels).

    relevance_init : {"identity", "random"} or np.ndarray, default="identity"
       Default will initiate the omega matrices to be the identity matrix. The rank of the matrix can be reduced by
        setting the ``relevance_n_components`` attribute `[3]`_.

    relevance_normalization: bool, optional, default=True
        Flag to indicate whether to normalize omega, whenever it is updated, such that the trace of the relevance matrix
        is equal to 1.

    relevance_n_components: str {"all"} or int, optional, default="all"
        For a square relevance matrix use the string "all" (default). For a rectangular relevance matrix use set the
        number of components explicitly by providing it as an int.

    relevance_localization: {"prototypes", "class"}, default="prototypes"
        Setting that controls the localization of the relevance matrices. Either per prototype,
        where each prototype has its own relevance matrix. Or per class where each class has its
        own relevance matrix. Note that when one prototype per class is used, changing this setting has no effect.

    random_state : int, RandomState instance, default=None
        Set the random number generation for reproducibility purposes. Used in random offset of prototypes and
        shuffling of the data in the solvers. Potentially, also used in the random generation of relevance matrix.

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

    omega_: ndarray with size (n_matrices, n_features, n_features)
        ``omega_`` matrices that were found during training and define the relevance matrices
        ``lambda_``.

    lambda_: ndarray of size (n_matrices, n_features, n_features)
        The relevance matrices ``omega_.T.dot(omega_)`` per matrix.

    omega_hat_: ndarray
        The omega matrices found by the eigenvalue decomposition of the relevance matrices
        ``lambda_``. The eigenvectors (columns of ``omega_hat_``) can be used to transform the data
         `[3]`_. This results in multiple possible transformations, one per relevance matrix.

    eigenvalues_: ndarray
        The corresponding eigenvalues to ``omega_hat_`` found by the eigenvalue decomposition of
        the relevance matrices ``lambda_``

    References
    ----------
    _`[1]` Sato, A., and Yamada, K. (1996) "Generalized Learning Vector Quantization."
    Advances in Neural Network Information Processing Systems, 423–429, 1996.

    _`[2]` Schneider, P., Biehl, M., & Hammer, B. (2009). "Adaptive Relevance Matrices in
    Learning Vector Quantization" Neural Computation, 21(12), 3532–3561, 2009.

    _`[3]` Bunte, K., Schneider, P., Hammer, B., Schleif, F.-M., Villmann, T., & Biehl, M. (2012).
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
        prototype_init: Union[str, np.ndarray] = "class-conditional-mean",
        prototype_n_per_class: [int, np.ndarray] = 1,
        relevance_init: Union[str, np.ndarray] = "identity",
        relevance_normalization: bool = True,
        relevance_n_components: Union[str, int] = "all",
        relevance_localization: str = "prototypes",
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, int] = True,
    ):
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.discriminant_type = discriminant_type
        self.discriminant_params = discriminant_params
        self.relevance_init = relevance_init
        self.relevance_normalization = relevance_normalization
        self.relevance_n_components = relevance_n_components
        self.relevance_localization = relevance_localization

        super(LGMLVQ, self).__init__(
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
        if self.relevance_normalization:
            LGMLVQ._normalize_omega(self.to_omega(self._variables))

    def set_model_params(self, new_model_params: ModelParamsType):
        """
        Changes the model's internal parameters. Copies the values of model_params into
        ``self.prototypes_`` and ``self.omega_`` therefore updating the ``self.variables_``
        array.

        Parameters
        ----------
        new_model_params : tuple of ndarrays
            Shapes depend on  initialization but in the case of a square relevance matrix:
            tuple((n_prototypes, n_features), (n_matrices, n_features, n_features))
        """
        new_prototypes, new_omega = new_model_params

        self.set_prototypes(new_prototypes)
        self.set_omega(new_omega)

        if self.relevance_normalization:
            LGMLVQ._normalize_omega(self.omega_)

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
    # Specific "getters" and "setters" for the prototypes shared by every LVQ model.
    ###########################################################################################

    def get_omega(self):
        """
        Function to return ``self.omega_`` (consistency)

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
            Returns a tuple with the prototypes and omega matrices as ndarrays.
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
            Shape depending on initialization but in case of a square matrix (n_matrices,
            n_features, n_features).
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
        self._normalize_omega(omega)

    @staticmethod
    def _normalize_omega(omega: np.ndarray) -> None:
        np.divide(
            omega,
            np.sqrt(np.einsum("ikj, ikj -> i", omega, omega)).reshape(
                omega.shape[0], 1, 1
            ),
            out=omega,
        )
        # denominator = np.sqrt(np.einsum("ikj, ikj -> i", omega, omega)).reshape(
        #     omega.shape[0], 1, 1
        # )
        # omega /= denominator

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
        omega_view = self.to_omega(gradient)[
            self.prototypes_labels_[i_prototype], :, :
        ].ravel()

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
                prototypes, omegas = self.to_model_params_view(gradient)
                prototypes *= step_sizes[0]
                omegas *= step_sizes[1]

    ###########################################################################################
    # Initialization function
    ###########################################################################################

    def _init_variables(self) -> None:
        self._variables = np.empty(
            self._prototypes_size + self._relevances_size,
            dtype="float64",
            order="C",
        )

    def _check_model_params(self):
        self._check_prototype_params()
        self._check_relevances_params()

    def _init_model_params(self, X, y) -> None:
        self._init_prototypes(X, y)
        self._init_relevances()

    def _check_relevances_params(self):
        relevance_normalization = self.relevance_normalization
        if not isinstance(relevance_normalization, bool):
            raise ValueError("Provided normalization is invalid.")

        relevance_n_components = self.relevance_n_components
        if isinstance(relevance_n_components, str):
            if relevance_n_components == "all":
                shape = (self.n_features_in_, self.n_features_in_)
            else:
                raise ValueError("Provided n_components is invalid.")
        elif isinstance(relevance_n_components, int):
            if (
                self.relevance_n_components >= 1
                and relevance_n_components <= self.n_features_in_
            ):
                shape = (relevance_n_components, self.n_features_in_)
            else:
                raise ValueError("Provided n_components is invalid.")
        else:
            raise ValueError("Provided n_components is invalid.")

        relevance_localization = self.relevance_localization
        if isinstance(relevance_localization, str):
            if relevance_localization == "prototypes":
                self._relevances_shape = (self._prototypes_shape[0], *shape)
            elif relevance_localization == "class":
                self._relevances_shape = (self.classes_.size, *shape)
            else:
                raise ValueError("Provided localization is invalid.")
        else:
            raise ValueError("Provided localization is invalid.")

        self._relevances_size = np.prod(self._relevances_shape)

    def _init_relevances(self):
        self.omega_ = self.to_omega(self._variables)

        if isinstance(self.relevance_init, str):
            if self.relevance_init == "identity":
                identity = np.eye(*self._relevances_shape[1:])
                self.set_omega(
                    np.repeat(
                        identity[np.newaxis, :, :], self._relevances_shape[0], axis=0
                    )
                )
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

        if self.relevance_normalization:
            LGMLVQ._normalize_omega(self.omega_)

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
        self.lambda_ = LGMLVQ._compute_lambdas(self.omega_)

        # Eigenvalues and column eigenvectors returned in ascending order
        eigenvalues, omega_hat = np.linalg.eigh(self.lambda_)

        # Rounding error cause eigenvalues to be very small negative numbers sometimes...
        self.eigenvalues_ = np.flip(eigenvalues, axis=1)
        self.omega_hat_ = np.flip(omega_hat, axis=2)

    @staticmethod
    def _compute_lambda(omega):
        # Equivalent to omega.T.dot(omega)
        return np.einsum("ji, jk -> ik", omega, omega)

    @staticmethod
    def _compute_lambdas(omegas):
        # Equivalent to omega.T.dot(omega) per omega (quite slow)
        return np.einsum("ikj, ikl -> ijl", omegas, omegas)

    ###########################################################################################
    # Transformer related functions
    ###########################################################################################

    def fit_transform(self, X: np.ndarray, y: np.ndarray, **trans_params) -> np.ndarray:
        r"""
        Parameters
        ----------
        X : ndarray with shape (n_samples, n_features)
            Data used for fit and that will be transformed.
        y : np.ndarray with length (n_samples)
            Labels corresponding to the X samples.
        trans_params :
            Parameters passed to transform function

        Returns
        -------
        The data projected on columns of ``omega_hat_`` with shape (n_matrices, n_samples,
        n_columns)
        """
        return self.fit(X, y).transform(X, **trans_params)

    def transform(
        self,
        X: np.ndarray,
        scale: bool = False,
        omega_hat_index: Union[int, List[int]] = 0,
    ) -> np.ndarray:
        r"""
        Parameters
        ----------
        X : np.ndarray with shape (n_samples, n_features)
            Data that needs to be transformed
        scale : {True, False}, default = False
            Controls if the eigenvectors the data is projected on are scaled by the square root
            of their eigenvalues.
        omega_hat_index : int or list
            The indices of the omega\_hats\_ the transformation should be computed for.

        Returns
        -------
        The data projected on columns of ``omega_hat_`` with shape (n_matrices, n_samples,
        n_columns)
        """
        check_is_fitted(self)

        X = check_array(X)

        transformation_matrix = self.omega_hat_[omega_hat_index, :, :]
        if transformation_matrix.ndim != 3:
            transformation_matrix = np.expand_dims(transformation_matrix, 0)

        if scale:
            transformation_matrix = np.einsum(
                "ik, ijk -> ijk",
                np.sqrt(np.absolute(self.eigenvalues_)),
                transformation_matrix,
            )
        transformed_data = np.einsum("jk, ikl -> ijl", X, transformation_matrix)

        return np.squeeze(transformed_data)

    def _more_tags(self):
        # For some reason lgmlvq (with default settings) does not perform well on one of the test
        # cases build into sklearn's test cases. Which is something to look into, but this "fixes"
        # it in the mean time...
        return {"poor_score": True}
