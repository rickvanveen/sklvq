from abc import ABC, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted, check_array

from .. import distances
from .. import solvers
from .._utils import init_class

from ..distances._base import DistanceBaseClass
from ..solvers._base import SolverBaseClass
from ..objectives._generalized_learning_objective import GeneralizedLearningObjective

from typing import Tuple, Union, List

ModelParamsType = np.ndarray

_PROTOTYPES_PARAMS_DEFAULTS = {"prototypes_per_class": 1}


class LVQBaseClass(ABC, BaseEstimator, ClassifierMixin):
    prototypes_: np.ndarray
    _distance: Union[DistanceBaseClass, object]
    _objective: GeneralizedLearningObjective
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
        prototype_params: dict = None,
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
        prototype_params = self._init_parameter_params(
            prototype_params, _PROTOTYPES_PARAMS_DEFAULTS
        )
        self.prototype_params = prototype_params

        self.random_state = random_state
        self.force_all_finite = force_all_finite

    @staticmethod
    def _init_parameter_params(parameter_params, parameter_params_defaults):
        if parameter_params is None:
            return parameter_params_defaults

        for key in parameter_params_defaults.keys():
            if key not in parameter_params:
                parameter_params[key] = parameter_params_defaults[key]

        return parameter_params

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    def set_variables(self, new_variables: np.ndarray) -> None:
        """
        Will modify the variables stored by the model (self) using numpy copyto.

        Parameters
        ----------
        new_variables : ndarray
            1d numpy array that contains all the model parameters in continuous memory

        """
        np.copyto(self._variables, new_variables)

    def get_variables(self) -> np.ndarray:
        """
        Consistency/convenience function.

        Returns
        -------
            variables : ndarray
                returns the models variables array.

        """
        return self._variables

    @abstractmethod
    def set_model_params(self, model_params: Union[tuple, np.ndarray]):
        """
        Changes the model's internal parameters.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def get_model_params(self) -> Union[tuple, np.ndarray]:
        """

        Returns
        -------
        ndarray or tuple
            In the simplest case returns only the prototypes as ndarray. Other models may include
            multiple parameters then they are stored in a tuple.

        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Specific "getters" and "setters" for the prototypes shared by every LVQ model.
    ###########################################################################################

    def set_prototypes(self, prototypes: np.ndarray) -> None:
        """
        Will copy the new prototypes into the model's prototypes, which is a view into the
        variables array and therefore will alter it.

        Parameters
        ----------
        prototypes : ndarray of shape (n_prototypes, n_features)
            The new prototypes the model should store.
        """
        np.copyto(self.prototypes_, prototypes)

    def get_prototypes(self) -> np.ndarray:
        """
        Convenience/consistency function. Only works after fitting the model and the prototypes
        have been initialized.

        Returns
        -------
            prototypes : ndarray of shape (n_prototypes, n_features)
                Returns a view of the shape specified above into the model's variables array.

        """
        return self.prototypes_

    ###########################################################################################
    # Functions to transform the 1D variables array to model parameters and back
    ###########################################################################################

    @abstractmethod
    def to_model_params(self, variables: np.ndarray) -> Union[tuple, np.ndarray]:
        """

        Parameters
        ----------
        variables : ndarray
            Single ndarray that stores the parameters in the order as given by the
            "to_variabes()" function

        Returns
        -------
        ndarray or tuple
            In the simplest case returns only the prototypes as ndarray. Other models may include
            multiple parameters then they are stored in a tuple.

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def to_prototypes(self, var_buffer: np.ndarray) -> np.ndarray:
        """
        Function that depends on a specific model. Should return a view (of the shape of the
        model's prototypes) into the provided variables buffer of the same size as the
        model's variables array.

        Parameters
        ----------
        var_buffer: ndarray
            1d array of the shame size as the model's variables array.

        Returns
        -------
            ndarray of shape (n_prototypes, n_features)

        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Solver Normalization functions
    ###########################################################################################

    @abstractmethod
    def normalize_variables(self, var_buffer: np.ndarray) -> None:
        """
        Should modify the as parameter given model_params in place.

        Parameters
        ----------
        model_params : ndarray or tuple
            Model parameters as provided by get_model_params()

        Returns
        -------
        ndarray or tuple
            Same shape and size as input, but normalized. How to normalize depends on model
            implementation.

        """
        raise NotImplementedError("You should implement this!")

    @staticmethod
    def _normalize_prototypes(prototypes: np.ndarray) -> None:
        """
        Modifies the prototypes argument.

        Parameters
        ----------
        prototypes : ndarray of shape (number of prototypes, number of dimensions)


        Returns
        -------
        ndarray of same shape as input
            Each prototype normalized according to w / norm(w)

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
    def add_partial_gradient(
        self, gradient, partial_gradient, i_prototype
    ) -> np.ndarray:
        """

        Parameters
        ----------
        gradient
        partial_gradient
        i_prototype

        Returns
        -------

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def multiply_variables(self, step_size, var_buffer: np.ndarray) -> None:
        """

        Parameters
        ----------
        step_size
        var_buffer

        Returns
        -------

        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Initialization function
    ###########################################################################################

    @abstractmethod
    def _init_variables(self):
        """
        Should initialize the variables, 1d numpy array to hold all model parameters. Should
        store these in self._variables.

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def _check_model_params(self):
        pass

    @abstractmethod
    def _init_model_params(self, X, y):
        """
        Depending on the model things such as self.prototypes_ should be initialized and set
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

    def _check_prototype_params(self, prototypes_per_class=1, **kwargs):
        if isinstance(prototypes_per_class, int):
            self._prototypes_shape = (
                prototypes_per_class * self.classes_.size,
                self.n_features_in_,
            )
        elif isinstance(prototypes_per_class, np.ndarray):
            if prototypes_per_class.size == self.classes_:
                self._prototypes_shape = (
                    np.prod(prototypes_per_class),
                    self.n_features_in_,
                )
            else:
                raise ValueError("Provided prototypes_per_class is invalid.")

        else:
            raise ValueError("Provided prototypes_per_class is invalid.")

        self._prototypes_size = np.prod(self._prototypes_shape)

    def _init_prototypes(
        self, X: np.ndarray, y: np.ndarray, prototypes_per_class=1,
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
            np.arange(0, self.classes_.size), prototypes_per_class
        )

        self.prototypes_ = self.to_prototypes(self._variables)

        if self.prototype_init == "class-conditional-mean":
            conditional_mean = _conditional_mean(self.prototypes_labels_, X, y)

            self.set_prototypes(
                conditional_mean
                + (1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape))
            )
        else:
            raise ValueError(
                "The provided value for the parameter 'prototypes' is invalid."
            )

    @abstractmethod
    def _init_objective(self) -> None:
        """
        Algorithm dependent.
        """
        raise NotImplementedError("You should implement this!")

    def _init_distance(self) -> None:
        """
        Initialize the distance function.
        """
        # Get the distance function and prepare parameters
        distance_params = {"force_all_finite": self.force_all_finite}
        distance_params.update(self.distance_params)

        distance_class = init_class(
            distances, self.distance_type, valid_class_types=self.valid_distances,
        )

        self._distance = distance_class(**distance_params)

    def _init_solver(self) -> None:
        """
        Should set the self._solver.
        """
        solver_class = init_class(solvers, self.solver_type, self.valid_solvers)
        self._solver = solver_class(self._objective, **self.solver_params)

    ###########################################################################################
    # Data and label validation
    ###########################################################################################

    def _validate_data_labels(
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
               views into self._variables.
            2. The distance function in self._distance
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

        # Initializes the 1D block of continuous memory.
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
        pass

    ###########################################################################################
    # Public API functions
    ###########################################################################################

    def fit(self, X: np.ndarray, y: np.ndarray):
        """ Fit function

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
        X, y_index = self._validate_data_labels(X, y)

        # Initialize random_state_ that should be used to perform any rng.
        self.random_state_ = check_random_state(self.random_state)

        # Before solve (handles initialization of things)
        self._before_fit(X, y_index)

        self._solver.solve(X, y_index, self)

        # After solve (handles initialization of things that can only be done after fit)
        self._after_fit(X, y_index)

        return self

    def decision_function(self, X: np.ndarray):
        """ Evaluates the decision function for the samples in X.

        Parameters
        ----------
        X : ndarray

        Returns
        -------

        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        X = check_array(X, force_all_finite=self.force_all_finite)

        # Of shape n_observations , n_prototypes
        distances = self._distance(X, self)

        # Allocation n_observations, n_classes
        min_distances = np.zeros((X.shape[0], self.classes_.size))

        # return n_observations, n_classes
        for i, c in enumerate(self.classes_):  # Correct?
            min_distances[:, i] = distances[:, self.prototypes_labels_ == i].min(axis=1)

        sum_min_distances = np.sum(min_distances, axis=1)

        decision_values = (1 - (min_distances / sum_min_distances[:, np.newaxis])) / 2

        # if binary then + for positive class and - for negative class
        if self.classes_.size == 2:
            return np.max(decision_values, axis=1) * (
                (np.argmax(decision_values, axis=1) * 2) - 1
            )

        return decision_values

    def predict(self, X: np.ndarray):
        """ Predict function

        Parameters
        ----------
        X

        Returns
        -------

        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        # X = self._validate_data(X, force_all_finite=self.force_all_finite)

        # TODO: Check the decision functions -> basically the score function from other lib?
        decision_values = self.decision_function(X)

        # Prototypes labels are indices of classes_
        if self.classes_.size == 2:
            return self.classes_[(decision_values > 0).astype(np.int)]

        return self.classes_[decision_values.argmax(axis=1)]


def _conditional_mean(p_labels: np.ndarray, data: np.ndarray, d_labels: np.ndarray):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array(
        [np.nanmean(data[p_label == d_labels, :], axis=0) for p_label in p_labels]
    )
