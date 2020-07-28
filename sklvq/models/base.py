from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import unique_labels, check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from sklvq.distances import DistanceBaseClass
from sklvq.solvers import SolverBaseClass
from sklvq.objectives import GeneralizedLearningObjective

from typing import Tuple, Union

ModelParamsType = np.ndarray

# Can be switched out by parameters to the models.
from sklvq import distances, solvers


class LVQBaseClass(ABC, BaseEstimator, ClassifierMixin):
    prototypes_: np.ndarray
    _distance: Union[DistanceBaseClass, object]
    _objective: GeneralizedLearningObjective

    def __init__(
        self,
        distance_type: Union[str, type] = "squared-euclidean",
        distance_params: dict = None,
        solver_type: Union[str, type] = "steepest-gradient-descent",
        solver_params: dict = None,
        initial_prototypes: Union[str, np.ndarray] = "class-conditional-mean",
        prototypes_per_class: Union[int, np.ndarray] = 1,
        random_state: Union[int, np.random.RandomState] = None,
        force_all_finite: Union[str, bool] = True,
    ):
        self.distance_type = distance_type
        self.distance_params = distance_params
        self.solver_type = solver_type
        self.solver_params = solver_params
        self.prototypes_per_class = prototypes_per_class
        self.initial_prototypes = initial_prototypes
        self.random_state = random_state
        self.force_all_finite = force_all_finite

    ###########################################################################################
    # The "Getter" and "Setter" that are used by the solvers to set and get model params.
    ###########################################################################################

    @abstractmethod
    def _set_model_params(self, model_params: Union[tuple, np.ndarray]) -> None:
        """
        Changes the model object's internal parameters.

        Parameters
        ----------
        model_params : ndarray or tuple
            In the simplest case can be only the prototypes as ndarray. Other models may include
            multiple parameters then they should be stored in a tuple.

        """
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def _get_model_params(self) -> Union[tuple, np.ndarray]:
        """

        Returns
        -------
        ndarray or tuple
            In the simplest case returns only the prototypes as ndarray. Other models may include
            multiple parameters then they are stored in a tuple.

        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # Functions to transform the 1D variables array to model parameters and back
    ###########################################################################################

    def _to_variables(self, model_params: Union[tuple, np.ndarray]) -> np.ndarray:
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
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def _to_params(self, variables: np.ndarray) -> Union[tuple, np.ndarray]:
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

    ###########################################################################################
    # Solver functions
    ###########################################################################################

    @abstractmethod
    def _normalize_params(
        self, model_params: Union[tuple, np.ndarray]
    ) -> Union[tuple, np.ndarray]:
        """

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

    ###########################################################################################
    # Initialization function
    ###########################################################################################

    @abstractmethod
    def _initialize(self, data: np.ndarray, y: np.ndarray) -> SolverBaseClass:
        """
        Functions should be implemented by every specific model. Must do the following two things
        in order to work:
            1. Must initialize the distance functions and store it in 'self._distance'
            2. Must initialize the solver and return it.

        Parameters
        ----------
        data : ndarray with shape (number of observations, number of dimensions)
            Provided for models which require the data for initialization.
        y : ndarray with size equal to the number of observations
            Provided for models which require the labels for initialization.

        Returns
        -------
        solver with as (implicit) base class SolverBaseClass.

        """
        raise NotImplementedError("You should implement this!")

    ###########################################################################################
    # LVQ specific functions
    ###########################################################################################

    @staticmethod
    def _normalize_prototypes(prototypes: np.ndarray) -> np.ndarray:
        """

        Parameters
        ----------
        prototypes : ndarray of shape (number of prototypes, number of dimensions)


        Returns
        -------
        ndarray of same shape as input
            Each prototype normalized according to w / norm(w)

        """
        return prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)

    def _initialize_prototypes(self, data: np.ndarray, y: np.ndarray):
        """
        Initialized the prototypes, with a small random offset, to the class conditional mean.

        Parameters
        ----------
        data : ndarray with shape (number of observations, number of dimensions)
        y : ndarray with size equal to the number of observations

        """
        if isinstance(self.initial_prototypes, np.ndarray):
            # TODO: Checks
            self.prototypes_ = self.initial_prototypes
        elif self.initial_prototypes == "class-conditional-mean":
            conditional_mean = _conditional_mean(self.prototypes_labels_, data, y)

            self.prototypes_ = conditional_mean + (
                1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape)
            )
        else:
            raise ValueError(
                "The provided value for the parameter 'prototypes' is invalid."
            )

    def _initialize_prototype_labels(self):
        # Assumes that if prototypes_per_class is an array this is meant to be the labeling for the
        # prototypes.
        if isinstance(self.prototypes_per_class, np.ndarray):
            self.prototypes_labels_ = self.prototypes_per_class
        # if the prototypes_per_class is a scalar assume equal number of prototypes per class and
        # initialize the labels. This is done within 'index' space and the actual labels can be
        # restored using self.classes_[self.prototype_labels_].
        elif np.isscalar(self.prototypes_per_class):
            self.prototypes_labels_ = np.repeat(
                np.arange(0, self.classes_.size), self.prototypes_per_class
            )
        else:
            raise ValueError(
                "The provided value for the parameter 'prototypes_per_class' is invalid."
            )

    ###########################################################################################
    # Data and label validation
    ###########################################################################################

    def _validate_data_labels(
        self, data: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Functions performs a series of check mostly by using sklearn util functions.
        Additionally, it transform the labels into indexes of unique labels, which are stored in
        self.classes_.

        Checks data and labels for consistent length, enforces X to be 2D and labels
        1D. By default, data is checked to be non-empty and containing only finite values. Standard
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
        data : ndarray of shape (number of observations, number of dimensions)
        labels : ndarray of size (number of observations)

        Returns
        -------
        data : ndarray with same shape (and values) as input
        labels : ndarray of indexes to self.classes_

        """
        # Check data
        data, labels = self._validate_data(
            data, labels, force_all_finite=self.force_all_finite
        )

        # Check classification targets
        check_classification_targets(labels)

        # Store unique provided labels in self.classes_ and transform the labels into a index
        # that can be used to reconstruct the original labels.
        self.classes_, labels = np.unique(labels, return_inverse=True)

        # Rais an error when the targets only contain a single unique class.
        if self.classes_.size <= 1:
            raise ValueError("Classifier can't train when only one class is present.")

        return data, labels

    def fit(self, data: np.ndarray, y: np.ndarray):
        """ Fit function

        Parameters
        ----------
        data
        y

        Returns
        -------

        """
        # Check data and check and transform labels.
        data, labels = self._validate_data_labels(data, y)

        # Initialize random_state_ that should be used to perform any rng.
        self.random_state_ = check_random_state(self.random_state)

        # Common LVQ steps -> move to specific model...
        # self._distance = distances.grab(self.distance_type, self.distance_params)

        # Initialize prototype labels stored in self.prototypes_labels_
        self._initialize_prototype_labels()

        # Using the now initialized (or checked custom prototype labels), we can initialize the
        # prototypes. Stored in self.prototypes_
        self._initialize_prototypes(data, labels)

        # Initialize algorithm specific stuff
        solver = self._initialize(data, labels)

        model = solver.solve(data, labels, self)

        # Useful for models such as GMLVQ, e.g., to compute lambda and it's eigenvectors/values
        # in order to transform the data.
        self._after_fit(data, labels)

        return model

    ###########################################################################################
    # After fit function
    ###########################################################################################

    def _after_fit(self, data: np.ndarray, y: np.ndarray):
        """
        Method that by default does nothing but can be used by methods that need to compute
        transformation matrices, such that this does not need to be checked or done everytime
        transform() is called (e.g., for GMLVQ and LGMLVQ)

        Parameters
        ----------
        data : ndarray with shape (number of observations, number of dimensions)
        y : ndarray with size equal to the number of observations
        """
        pass

    def decision_function(self, data: np.ndarray):
        """ Decision function

        Parameters
        ----------
        data

        Returns
        -------

        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = check_array(data, force_all_finite=self.force_all_finite)

        # Of shape n_observations , n_prototypes
        distances = self._distance(data, self)

        # Allocation n_observations, n_classes
        min_distances = np.zeros((data.shape[0], self.classes_.size))

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

    def predict(self, data: np.ndarray):
        """ Predict function

        Parameters
        ----------
        data

        Returns
        -------

        """
        # SciKit-learn list of checked params before predict
        check_is_fitted(self)

        # Input validation
        data = self._validate_data(data, force_all_finite=self.force_all_finite)

        decision_values = self.decision_function(data)

        # TODO: Reject option?
        # Prototypes labels are indices of classes_
        # return self.prototypes_labels_.take(self._distance(data, self).argmin(axis=1))
        if self.classes_.size == 2:
            return self.classes_[(decision_values > 0).astype(np.int)]

        return self.classes_[decision_values.argmax(axis=1)]


def _conditional_mean(p_labels: np.ndarray, data: np.ndarray, d_labels: np.ndarray):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array(
        [np.nanmean(data[p_label == d_labels, :], axis=0) for p_label in p_labels]
    )
