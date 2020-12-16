from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array

from .. import distances
from .. import solvers
from ..solvers import SolverBaseClass
from .._utils import init_class

from ..distances import DistanceBaseClass
from ..objectives import GeneralizedLearningObjective

from typing import Tuple, Union, List

ModelParamsType = np.ndarray


class LVQBaseClass(
    ABC, BaseEstimator, ClassifierMixin
):  # lgtm [py/conflicting-attributes]
    """Learning Vector Quantization base class

    Abstract class for implementing LVQ models. It provides abstract methods with
    expected call signatures.

    Provides a common interface to the solver and other function that require access to the
    models. Additionally, it implements a number of functions shared by the currently implemented
    LVQ variations.

    See also
    --------
    GLVQ, GMLVQ, LGMLVQ
    """

    # Public attributes
    prototypes_: np.ndarray

    # "Private" attributes
    _distance: Union[DistanceBaseClass, object]
    _objective: GeneralizedLearningObjective
    _solver: Union[SolverBaseClass, object]

    # Related to model parameters
    _prototypes_size: int
    _prototypes_shape: Tuple

    # _variables stores the  model parameters (e.g., prototypes) in 1D format. The
    # get_prototypes() and other model parameter functions should return a view into this 1D
    # array.  Likewise set_prototypes() overwrites the _variables array.
    _variables: np.ndarray

    def __init__(
        self,
        distance_type: Union[str, type] = "squared-euclidean",
        distance_params: dict = None,
        valid_distances: List[str] = None,
        solver_type: Union[str, type] = "steepest-gradient-descent",
        solver_params: dict = None,
        valid_solvers: List[str] = None,
        prototype_init: Union[str, np.ndarray] = "class-conditional-mean",
        prototype_n_per_class: int = 1,
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, bool] = True,
    ):
        self.distance_type = distance_type
        if distance_params is None:
            distance_params = {}
        self.distance_params = distance_params
        self.valid_distances = valid_distances

        self.solver_type = solver_type
        if solver_params is None:
            solver_params = {}
        self.solver_params = solver_params
        self.valid_solvers = valid_solvers

        self.prototype_init = prototype_init
        self.prototype_n_per_class = prototype_n_per_class

        self.random_state = random_state
        self.force_all_finite = force_all_finite

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    def get_variables(self) -> np.ndarray:
        r"""
        Returns the ``self._variables`` array that owns the memory allocated for the model
        parameters.

        Returns
        -------
        _variables : ndarray
            returns the model's _variables array.
        """
        return self._variables

    def set_variables(self, new_variables: np.ndarray) -> None:
        r"""
        Modifies the ``self._variables`` by copying the values of ``new_variables`` into the
        memory of ``self._variables``.

        Parameters
        ----------
        new_variables : ndarray
            1d numpy array that contains all the model parameters in continuous memory
        """
        np.copyto(self._variables, new_variables)

    @abstractmethod
    def get_model_params(self) -> Union[tuple, np.ndarray]:
        r"""
        Should return a view or tuple of views (in correct shape) of the model's parameters.
        Implementation depends on specific model as model parameters may differ per model.

        Returns
        -------
        ndarray or tuple
            View or tuple of views of the model's parameters.
        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def set_model_params(self, new_model_params: Union[tuple, np.ndarray]):
        r"""
        Should modify the ``self._variables`` array. Accepts the new_model_params in the shape of
        the model's parameters, e.g., prototypes or (prototypes, relevance_matrix).

        Always needs to be all the model parameters, can not be used for partial updates.

        Parameters
        ----------
        new_model_params : ndarray or tuple
            Array or tuple of arrays of the new model's parameters.
        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Specific "getters" and "setters" for the prototypes shared by every LVQ model.
    ###########################################################################################

    def get_prototypes(self) -> np.ndarray:
        r"""
        Return a view into ``self._variables`` of the the shape of the prototypes (n_prototypes,
        n_features). At the moment only consistency function, does not actually  create the shape
        and only  works after ``self.prototypes_`` has been set.

        Returns
        -------
        prototypes : ndarray of shape (n_prototypes, n_features)
            View into ``self._variables`` with shape specified above.
        """
        return self.prototypes_

    def set_prototypes(self, new_prototypes: np.ndarray) -> None:
        r"""
        Accepts a new_prototypes array with the same shape as ``self.prototypes_`` and overwrites
        the ``self._variables`` array by copying the values of the new_prototypes.

        Parameters
        ----------
        new_prototypes : ndarray of shape (n_prototypes, n_features)
            The new prototypes the model should store.
        """
        np.copyto(self.prototypes_, new_prototypes)

    ###########################################################################################
    # Functions to transform the 1D variables array to model parameters and back
    ###########################################################################################

    @abstractmethod
    def to_model_params_view(self, var_buffer: np.ndarray) -> Union[tuple, np.ndarray]:
        r"""
        Should return a single view into the var_buffer or a tuple of views. This depends on the
        model and its parameters.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        ndarray or tuple
            Should return a view or tuple of views of the model parameters in appropriate shapes.
        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def to_prototypes_view(self, var_buffer: np.ndarray) -> np.ndarray:
        r"""
        Should return the prototypes from the provided var_buffer. I.e., it selects/views the
        appropriate part of memory and reshapes it.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.

        Returns
        -------
        ndarray of shape (n_prototypes, n_features)
            View into the var_buffer.
        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Solver Normalization functions
    ###########################################################################################

    @abstractmethod
    def normalize_variables(self, var_buffer: np.ndarray) -> None:
        r"""
        Should modify the var_buffer as if it was the variables array provided
        by ``get_variables()``.

        Parameters
        ----------
        var_buffer : ndarray
            Array with the same size as the model's variables array as returned
            by ``get_variables()``.


        Returns
        -------
        ndarray or tuple
            Same shape and size as input, but normalized. How to normalize depends on model
            implementation.
        """
        raise NotImplementedError("You should implement this!")

    @staticmethod
    def _normalize_prototypes(prototypes: np.ndarray) -> None:
        r"""
        Normalizes the provided prototypes array, i.e., it writes to the same memory. Performs
        the following normalization step for each prototype :math:`\mathbf{w_i}`:

        ..math::
             \mathbf{w}_i / || \mathbf{w}_i ||

        Parameters
        ----------
        prototypes : ndarray of shape (n_prototypes, n_features)
            To be normalized prototypes.

        Returns
        -------
        ndarray of same shape as input
            Normalized prototypes.
        """
        np.divide(
            prototypes,
            np.linalg.norm(prototypes, axis=1, keepdims=True),
            out=prototypes,
        )

    ###########################################################################################
    # Solver helper functions
    ###########################################################################################

    @abstractmethod
    def add_partial_gradient(self, gradient, partial_gradient, i_prototype) -> None:
        r"""
        To increase performance, the distance gradient methods (should) return only the relevant values.
        I.e., the gradient of the prototype i_prototype and potentially other parameters linked to
        this prototype. This partial gradient needs to added (overwrite) to the correct parts of
        the actual gradient and this is what this function should do.

        Parameters
        ----------
        gradient : ndarray
            Same shape as the ``get_variables()`` would return.

        partial_gradient : ndarray
            1d array containing the partial gradient.

        i_prototype : int
            The index of the prototype to which the partial gradient was  computed.
        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def mul_step_size(
        self, step_size: Union[float, np.ndarray], gradient: np.ndarray
    ) -> None:
        r"""
        Should multiply the provided gradient with the provided step size and overwrite the
        values in ``gradient``.  Depending on the ``step_size`` being a float or array different
        step sizes are used for different model parameters (which also depends on the model if
        there are more then only prototypes)

        Parameters
        ----------
        step_size : float or ndarray
            The scalar or list of values containing the step sizes.
        gradient : ndarray
            Same shape as the ``get_variables()`` would return.
        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Initialization function
    ###########################################################################################

    @abstractmethod
    def _init_variables(self):
        r"""
        Should initialize the variables, 1d numpy array to hold all model parameters. Should
        store these in self._variables.
        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def _check_model_params(self):
        r"""
        Should check the model parameters. I.e., call check_prototype_params with parameters and
        other model parameters that there might be.
        """

    @abstractmethod
    def _init_model_params(self, X, y):
        r"""
        Depending on the model, things such as self.prototypes_ should be initialized and set
        using the set methods or one should make sure the parameters are views into the variables
        array, such that variables array is changed as well.

        Parameters
        ----------
        X : ndarray, with shape (n_samples, n_features)
            The X
        y : ndarray, with shape (n_samples)
            The labels
        """
        raise NotImplementedError("You should implement this!")

    def _check_prototype_params(self):
        """
        Check prototype params, i.e., if the prototypes_per_class is set correctly.
        Additionally, it sets the size and shape of the prototypes such that these can be used
        for the creation of the ``self._variables`` and view ``self.prototypes_``.
        """
        prototype_n_per_class = self.prototype_n_per_class

        if isinstance(prototype_n_per_class, int):
            self._prototypes_shape = (
                prototype_n_per_class * self.classes_.size,
                self.n_features_in_,
            )
        elif isinstance(prototype_n_per_class, np.ndarray):
            if prototype_n_per_class.size != self.classes_.size:
                raise ValueError(
                    "Expected the number protoypes_per_class (size = {}) to have a number of elements "
                    "equal to the number of classes (size = {}).",
                    prototype_n_per_class.size,
                    self.classes_.size,
                )

            if np.any(prototype_n_per_class <= 0.0):
                raise ValueError(
                    "Prototypes_per_class ({}) cannot contain any values less than or equal to zero.",
                    prototype_n_per_class,
                )

            self._prototypes_shape = (
                np.sum(prototype_n_per_class),
                self.n_features_in_,
            )
        else:
            raise ValueError(
                "Expected prototypes_per_class to be either of type int or np.ndarray, but got type: {}",
                type(prototype_n_per_class),
            )

        self._prototypes_size = np.prod(self._prototypes_shape)

    def _init_prototypes(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> None:
        """
        Initialized the prototypes, with a small random offset, to the class conditional mean.
        To be used in the _initialize_parameters function.

        Parameters
        ----------
        X : ndarray with shape (number of observations, number of dimensions)
        y : ndarray with size equal to the number of observations
        """
        self.prototypes_labels_ = np.repeat(
            np.arange(0, self.classes_.size), self.prototype_n_per_class
        )

        # Sets initial value for prototypes....
        self.prototypes_ = self.to_prototypes_view(self._variables)

        if self.prototype_init == "class-conditional-mean":
            conditional_mean = _conditional_mean(self.prototypes_labels_, X, y)

            self.set_prototypes(
                conditional_mean
                + (1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape))
            )
        else:
            raise ValueError(
                "The provided value for the parameter 'prototype_init' is invalid."
            )

    @abstractmethod
    def _init_objective(self) -> None:
        """
        Should initialize the ``self._objective``. Depends on the algorithm.
        """
        raise NotImplementedError("You should implement this!")

    def _init_distance(self) -> None:
        """
        Initializes the ``self.distance``.
        """
        distance_class = init_class(
            distances,
            self.distance_type,
            valid_class_types=self.valid_distances,
        )

        self._distance = distance_class(**self.distance_params)

    def _init_solver(self) -> None:
        """
        Should initialize the ``self._solver``. Depends on the algorithm.
        """
        solver_class = init_class(solvers, self.solver_type, self.valid_solvers)
        self._solver = solver_class(self._objective, **self.solver_params)

    ###########################################################################################
    # Data and label validation
    ###########################################################################################

    def _check_data_and_labels(
        self, X: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Functions performs a series of check mostly by using sklearn util functions.
        Additionally, it transform the labels into indexes of unique labels, which are stored in
        self.classes_.

        Checks X and labels for consistent length, enforces X to be 2D and labels
        1D. By default, X is checked to be non-empty and containing only finite values. Standard
        input checks are also applied to labels, such as checking that labels does not have
        np.nan or np.inf targets.
            - sklearn check_X_y()

        Ensure that target labels are of a non-regression type.
        Only the following target types (as defined in sklearn.utils.multiclass.type_of_target)
        are allowed:
            'binary', 'multiclass', 'multiclass-multioutput',
            'multilabel-indicator', 'multilabel-sequences'.
            - sklearn check_classification_targets()

        Parameters
        ----------
        X : ndarray of shape (number of observations, number of dimensions)
        labels : ndarray of size (number of observations)

        Returns
        -------
        X : ndarray with same shape (and values) as input
        labels : ndarray of indexes to self.classes_
        """
        # Check X
        X, labels = self._validate_data(
            X, labels, force_all_finite=self.force_all_finite
        )

        # Check classification targets
        check_classification_targets(labels)

        # Store unique provided labels in self.classes_ and transform the labels into a index
        # that can be used to reconstruct the original labels.
        self.classes_, labels = np.unique(labels, return_inverse=True)

        # Raise an error when the targets only contain a single unique class.
        if self.classes_.size <= 1:
            raise ValueError("Classifier can't train when only one class is present.")

        return X, labels

    ###########################################################################################
    # Before and after fit/solve function (initialization of things)
    ###########################################################################################

    def _before_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Should initialize:
            1. self._variables and algorithm specific parameters which should be
               views into self._variables (share memory).
            2. The distance function in self.distance
            3. The objective function in self._objective
            4. The solver function in self._solver

        Parameters
        ----------
        X : ndarray, with shape (n_samples, n_features)
            The X
        y : ndarray, with shape (n_samples)
            The labels
        """
        self._check_model_params()

        # Initializes the 1D block of continuous memory to hold the model params.
        self._init_variables()

        # Initialize algorithm specific parameters.
        self._init_model_params(X, y)

        self._init_distance()

        self._init_objective()

        self._init_solver()

    def _after_fit(self, X: np.ndarray, y: np.ndarray):
        """
        Method that by default does nothing but can be used by methods that need to compute
        transformation matrices, such that this does not need to be checked or done everytime
        transform() is called (e.g., for GMLVQ and LGMLVQ)

        Parameters
        ----------
        X : ndarray with shape (number of observations, number of dimensions)
        y : ndarray with size equal to the number of observations
        """

    ###########################################################################################
    # Public API functions
    ###########################################################################################

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit function that provides the general implementation of the LVQ algorithms. It checks the data, calls
         before_fit method, calls the solve method of the solver, and the after_fit method.

        Parameters
        ----------
        X : ndarray of shape (number of observations, number of dimensions)
        y : ndarray of size (number of observations)

        Returns
        -------
        self
            The trained model
        """
        # Check X and check and transform labels.
        X, y_index = self._check_data_and_labels(X, y)

        # Initialize random_state_ that should be used to perform any rng.
        self.random_state_ = check_random_state(self.random_state)

        # Before solve (handles initialization of things)
        self._before_fit(X, y_index)

        self._solver.solve(X, y_index, self)

        # After solve (handles initialization of things that can only be done after fit)
        self._after_fit(X, y_index)

        return self

    def _multiclass_decision_function(self, X: np.ndarray):
        """
        Computes the decision values and returns shape (n_observations, n_classes). The values
        are constructed by computing: the distance between an observation and the prototype with
        a different label minus the distance between that observation and the closest prototype
        with the same label, where "the same" is defined to be the index of the column the value
        is being computed for.

        Parameters
        ----------
        X : ndarray
            The data.

        Returns
        -------
        ndarray of shape (n_observations, n_classes)
        """
        # Of shape n_observations , n_prototypes
        distances = self._distance(X, self)

        # Allocation n_observations, n_classes
        decision_values = np.zeros((X.shape[0], self.classes_.size))

        # return n_observations, n_classes
        for i, _ in enumerate(self.classes_):
            decision_values[:, i] = distances[:, self.prototypes_labels_ != i].min(
                axis=1
            ) - distances[:, self.prototypes_labels_ == i].min(axis=1)

        return decision_values

    def decision_function(self, X: np.ndarray):
        """
        Evaluates the decision function for the samples in X.
        Shape for binary class is (n_observations,) with the decision values for the "greater"
        class. In the multiclass case it returns decision values for each class and therefore has
        the shape (n_observations, n_classes).

        Parameters
        ----------
        X : ndarray
            The data.

        Returns
        -------
        decision_values : ndarray
            Binary case shape is (n_observations,) and the multiclass case (n_observations,
            n_classes)
        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        X = check_array(X, force_all_finite=self.force_all_finite)

        decision_values = self._multiclass_decision_function(X)

        if self.classes_.size == 2:
            # The decision function needs to return (n_samples,) for the "greater" class.
            return decision_values[:, 1]

        # else it should  return (n_samples, n_classes)
        return decision_values

    def predict_proba(self, X: np.ndarray):
        """
        Parameters
        ----------
        X  : ndarray
            The data.

        Returns
        -------
        confidence_scores : ndarray of shape (n_observations, n_classes)
        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        X = check_array(X, force_all_finite=self.force_all_finite)

        # Between -1  and 1
        decision_values = self._multiclass_decision_function(X)

        # Softmax function (keeps the same scipy.stats.rankdata)
        # Very  arbitrary  0.01, which also might not always work?
        exp_decision_values = np.exp(0.01 * decision_values)

        return exp_decision_values / np.sum(exp_decision_values, axis=1)[:, np.newaxis]

    def predict(self, X: np.ndarray):
        """Predict function

        The decision is made for the label of the prototype with the minimum decision value,
        as provided by the ``decision_function()``.

        Parameters
        ----------
         X  : ndarray
            The data.

        Returns
        -------
        ndarray of shape (n_observations)
            Returns the predicted labels.
        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        # X = self._validate_data(X, force_all_finite=self.force_all_finite)
        decision_values = self.decision_function(X)

        if self.classes_.size == 2:
            return self.classes_[(decision_values > 0).astype(np.int)]

        # Lower value is the closest prototype.
        return self.classes_[decision_values.argmax(axis=1)]


def _conditional_mean(p_labels: np.ndarray, data: np.ndarray, d_labels: np.ndarray):
    """ Implements the conditional mean (ignoring nan values), i.e., mean per class"""
    return np.array(
        [np.nanmean(data[p_label == d_labels, :], axis=0) for p_label in p_labels]
    )
