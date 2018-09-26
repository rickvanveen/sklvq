
# Template and strategy design patterns.... Factory?
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels, check_classification_targets

import scipy as sp
import numpy as np

# TODO: Move all factories to init files??
# TODO: Datastructures for RelevanceMatrix and Prototype(s) E.g.,

def _conditional_mean(p_labels, data, d_labels):
    """ Implements the conditional mean, i.e., mean per class"""
    return np.array([np.mean(data[p_label == d_labels, :], axis=0)
                     for p_label in p_labels])


# Template (Context)
class LVQClassifier(ABC, BaseEstimator, ClassifierMixin):

    # But here I can have logic... because it's not a sklearn estimator?
    # Cannot change the value of the properties given in init...
    def __init__(self, prototypes_per_class, random_state):
        self.prototypes_per_class = prototypes_per_class
        self.random_state = random_state

    @abstractmethod
    def initialize(self, data, y):
        raise NotImplementedError("You should implement this! Must accept (data, y)"
                                  " and return Objective and Solver objects")

    # TODO: could also be class functions things that can be extended by user by providing another class same for omega.
    def init_prototypes(self, data, y):
        conditional_mean = _conditional_mean(self.prototypes_labels, data, y)
        return conditional_mean + (1e-4 * self.random_state_.uniform(-1, 1, conditional_mean.shape))

    def validate(self, data, y):
        # SciKit-learn required check
        data, labels = check_X_y(data, y)

        # SciKit-learn required check
        check_classification_targets(y)

        return labels

    # TODO: Change to _fit_solver1 and _fit_solver2 and call in the specific the right fit
    def fit(self, data, y):
        # Validate SciKit-learn required stuff...
        labels = self.validate(data, y)

        # SciKit-learn way of doing random stuff...
        self.random_state_ = check_random_state(self.random_state)

        # Common LVQ steps
        # I guess it's save to say that LVQ always needs to have initialized prototypes/prototype_labels
        if np.isscalar(self.prototypes_per_class):
            self.prototypes_labels = np.repeat(unique_labels(labels), self.prototypes_per_class)

        self.prototypes_ = self.init_prototypes(data, labels)
        # Initialize algorithm specific stuff
        solver, variables, objective_args = self.initialize(data, labels)

        variables = solver.solve(variables, objective_args)

        num_features = data.shape[1]
        num_prototypes = self.prototypes_labels.size

        self.prototypes_ = variables.reshape(num_prototypes, num_features)

        return self

    def predict(self, data):
        # check_is_fitted(self, ['prototypes_', 'prototype_labels', 'classes_'])

        # Input validation
        data = check_array(data)

        distance = DistanceFactory.create(self.distance)
        # TODO: Very simple version which probably only works nicely in the examples
        return self.prototypes_labels.take(distance(data, self.prototypes_).argmin(axis=1))


# Template (Context Implementation)
class GLVQClassifier(LVQClassifier):

    def __init__(self, distance='sqeuclidean', solver='l-bfgs-b', scaling='identity', beta=None, verbose=False,
                 prototypes_per_class=1, random_state=None):
        self.distance = distance
        self.solver = solver
        self.scaling = scaling
        self.beta = beta
        self.verbose = verbose
        super(GLVQClassifier, self).__init__(prototypes_per_class, random_state)

    def initialize(self, data, labels):
        """ . """
        # Here add inits for RelativeDistanceObjective... self.distance, self.scaling...
        distance = DistanceFactory.create(self.distance)
        scaling = ScalingFactory.create(self.scaling)
        scaling.beta = self.beta

        objective = RelativeDistanceObjective(distance=distance, scaling=scaling,
                                              prototypes_shape=self.prototypes_.shape)

        solver = SolverFactory.create(self.solver)
        solver.objective = objective

        variables = self.prototypes_.ravel()
        objective_args = (self.prototypes_labels, data, labels)

        return solver, variables, objective_args


class GMLVQClassifier(LVQClassifier):

    def __init__(self, distance='sqeuclidean', solver='l-bfgs-b', scaling='identity', beta=None, verbose=False,
                 omega='identity', omega_shape=None,  prototypes_per_class=1, random_state=None):
        self.distance = distance
        self.solver = solver
        self.scaling = scaling
        self.beta = beta
        self.verbose = verbose
        self.omega = omega
        self.omega_shape = omega_shape
        super(GMLVQClassifier, self).__init__(prototypes_per_class, random_state)

    def _init_relevance(self, data):
        pass

    def initialize(self, data, y):

        # objective = RelevanceRelativeDistance()

        solver = SolverFactory.create(self.solver)
        solver.objective = object

        variables = self.prototypes_.ravel()

        objective_args = ()

        return solver, variables, objective_args


# ---------------------------------------------------------------------------------------------------------------------

def _find_min(indices, distances):
    """ Helper function to find the minimum distance and the index of this distance. """
    dist_temp = np.where(indices, distances, np.inf)
    return dist_temp.min(axis=1), dist_temp.argmin(axis=1)


# Strategy
class AbstractObjective(ABC):

    def __init__(self, distance):
        self.distance = distance

    @abstractmethod
    def __call__(self, variables, prototype_labels, data, labels):
        raise NotImplementedError("You should implement this!")

    def compute_distance(self, data, labels, prototypes, prototype_labels):
        """ Computes the distances between each prototype and each observation and finds all indices where the smallest
        distance is that of the prototype with the same label and with a different label. """

        distances = self.distance(data, prototypes)

        ii_same = np.transpose(np.array([labels == prototype_label for prototype_label in prototype_labels]))
        ii_diff = ~ii_same

        dist_same, i_dist_same = _find_min(ii_same, distances)
        dist_diff, i_dist_diff = _find_min(ii_diff, distances)

        return dist_same, dist_diff, i_dist_same, i_dist_diff


# GLVQ - mu(x) functions
def _relative_distance(dist_same, dist_diff):
    return (dist_same - dist_diff) / (dist_same + dist_diff)


# derivative mu(x) in glvq paper, same and diff are relative to the currents prototype's label
def _relative_distance_grad(dist_same, dist_diff):
    return 2 * dist_diff / (dist_same + dist_diff) ** 2


# Strategy (Strategy Implementation)
class RelativeDistanceObjective(AbstractObjective):

    def __init__(self, distance=None, scaling=None, prototypes_shape=None):
        self.prototypes_shape = prototypes_shape
        self.scaling = scaling
        super(RelativeDistanceObjective, self).__init__(distance)

    def __call__(self, variables, prototype_labels, data, labels):
        num_prototypes = prototype_labels.size

        # Different for GMLVQ
        prototypes = variables.reshape(self.prototypes_shape)

        dist_same, dist_diff, i_dist_same, i_dist_diff = self.compute_distance(data, labels,
                                                                               prototypes, prototype_labels)

        gradient = np.zeros(prototypes.shape)

        for i_prototype in range(0, num_prototypes):
            ii_same = i_prototype == i_dist_same
            ii_diff = i_prototype == i_dist_diff

            # f'(mu(x)) * (2 * d_2(x) / (d_1(x) + d_2(x))^2)
            relative_dist_same = (
                        self.scaling.gradient(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) *
                        _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
            relative_dist_diff = (
                        self.scaling.gradient(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff])) *
                        _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))
            # -2 * (x - w)
            grad_same = self.distance.gradient(data[ii_same, :], prototypes[i_prototype, :])
            grad_diff = self.distance.gradient(data[ii_diff, :], prototypes[i_prototype, :])

            gradient[i_prototype, :] = (relative_dist_same.dot(grad_same) - relative_dist_diff.dot(grad_diff))

        return np.sum(self.scaling(_relative_distance(dist_same, dist_diff))), gradient.ravel()


class RelevanceRelativeDistanceObjective(AbstractObjective):

    def __init__(self, distance=None, scaling=None, prototypes_shape=None, omega_shape=None):
        self.prototypes_shape = prototypes_shape
        self.omega_shape = omega_shape
        self.scaling = scaling
        super(RelevanceRelativeDistanceObjective, self).__init__(distance)


    def __call__(self, variables, prototype_labels, data, labels):
        num_features = data.shape[1]
        num_prototypes = prototype_labels.size

        prototypes_variables = variables[:(num_prototypes * num_features)]
        prototypes = prototypes_variables.reshape(prototype_labels.size, num_features)

        # TODO: Works for Global but not Local matrices...
        omega_variables = variables[(num_prototypes * num_features):]
        omega = omega_variables.reshape(self.omega_shape)

        # Update omega for the distance object used in compute_distance
        self.distance.omega = omega

        dist_same, dist_diff, i_dist_same, i_dist_diff = self.compute_distance(data, labels,
                                                                               prototypes, prototype_labels)

        gradient = np.zeros(prototypes.shape)
        gradient_omega = np.zeros(omega.shape)

        for i_prototype in range(0, num_prototypes):
            ii_same = i_prototype == i_dist_same
            ii_diff = i_prototype == i_dist_diff

            # f'(mu(x)) * (2 * d_2(x) / (d_1(x) + d_2(x))^2)
            relative_dist_same = (
                        self.scaling.gradient(_relative_distance(dist_same[ii_same], dist_diff[ii_same])) *
                        _relative_distance_grad(dist_same[ii_same], dist_diff[ii_same]))
            relative_dist_diff = (
                        self.scaling.gradient(_relative_distance(dist_diff[ii_diff], dist_same[ii_diff])) *
                        _relative_distance_grad(dist_diff[ii_diff], dist_same[ii_diff]))

            # Gradient prototypes
            grad_same = self.distance.gradient(data[ii_same, :], prototypes[i_prototype, :])
            grad_diff = self.distance.gradient(data[ii_diff, :], prototypes[i_prototype, :])

            gradient[i_prototype, :] = (relative_dist_same.dot(grad_same) - relative_dist_diff.dot(grad_diff))

            # Gradient Omega


        return np.sum(self.scaling(_relative_distance(dist_same, dist_diff))), gradient.ravel()







# TODO: Not actually used...
class ObjectiveFactory():

    @staticmethod
    def create(objective_type):
        if objective_type == 'relative-distance':
            return RelativeDistanceObjective()
        else:
            print("Objective type does not exist")


# ---------------------------------------------------------------------------------------------------------------------
# Strategy
class AbstractSolver(ABC):

    def __init__(self, objective):
        self.objective = objective

    @abstractmethod
    def solve(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")


# Strategy (Strategy Implementation)
#  TODO: How should this work now... because it's an explicit solver and specific for the algorithm.
class StochasticSolver(AbstractSolver):

    def __init__(self):
        super(StochasticSolver, self).__init__()

    def solve(self, *args, **kwargs):
        print('Calling StochasticSolver.solve()')


# Strategy (Strategy Implementation)
class LBFGSBSolver(AbstractSolver):
    
    def __init__(self, objective=None):
        super(LBFGSBSolver, self).__init__(objective)

    def solve(self, variables, objective_args):
        result = sp.optimize.fmin_l_bfgs_b(self.objective, variables, args=objective_args)
        return result[0]


# TODO: this could read a config...
class SolverFactory:

    @staticmethod
    def create(solver_type):
        if solver_type == 'stochastic':
            return StochasticSolver()
        elif solver_type == 'l-bfgs-b':
            return LBFGSBSolver()
        else:
            print("Solver type does not exist")


# ---------------------------------------------------------------------------------------------------------------------
class AbstractDistance(ABC):

    @abstractmethod
    def __call__(self, data, prototypes):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, data, prototype):
        raise NotImplementedError("You should implement this!")


class SquaredEuclidean(AbstractDistance):
    
    def __call__(self, data, prototypes):
        return sp.spatial.distance.cdist(data, prototypes, 'sqeuclidean')

    def gradient(self, data, prototype):
        return -2 * (data - prototype)


class Euclidean(AbstractDistance):

    def __call__(self, data, prototypes):
        return sp.spatial.distance.cdist(data, prototypes, 'euclidean')

    def gradient(self, data, prototype):
        difference = data - prototype
        return (-1 * difference) / np.sqrt(np.sum(difference ** 2))


class RelevanceSquaredEuclidean(AbstractDistance):

    # init in ___init__ is annoying for stuff that changes all the time: distance.omega = new_omega; distance(data, prototypes) instead of just distance(data, prototypes, omega=omega)
    def __init__(self, omega=None):
        self.omega = omega

    def __call__(self, data, prototypes):
        return sp.spatial.distance.cdist(data, prototypes, 'mahanalobis', self.omega) ** 2

    def gradient(self, data, prototype):
        pass


class DistanceFactory:

    # TODO: In __init__.py and with dict {'sqeuclidean': 'SquaredEuclidean'}
    # TYPES = {'sqeuclidean': 'SquaredEuclidean',
    #                   'euclidean': 'Euclidean',
    #                   'releuclidean': 'RelevanceSquaredEuclidean'}

    @staticmethod
    def create(distance_type, *args, **kwargs):
        # try:
        #     distance_object = getattr(sys.module[__name__], DistanceFactory.TYPES[distance_type])
        if distance_type == 'sqeuclidean':
            return SquaredEuclidean(*args, **kwargs)
        if distance_type == 'euclidean':
            return Euclidean(*args, **kwargs)
        else:
            print("Distance type does not exist")

# ----------------------------------------------------------------------------------------------------------------------


# Defines a interface for the algorithms... every scaling function has a compute and compute_gradient...
# additional constant parameters can be set during initialisation...
class AbstractScaling(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")

    @abstractmethod
    def gradient(self, *args, **kwargs):
        raise NotImplementedError("You should implement this!")


class Identity(AbstractScaling):

    def __call__(self, x):
        return x

    def gradient(self, x):
        return 1


class Sigmoid(AbstractScaling):

    def __init__(self, beta=2):
        self.beta = beta

    def __call__(self, x):
        return 1 / (np.exp(-self.beta * x) + 1)

    def gradient(self, x):
        return (self.beta * np.exp(self.beta * x)) / (np.exp(self.beta * x) + 1) ** 2


class ScalingFactory:

    @staticmethod
    def create(scaling_type):
        if scaling_type == 'identity':
            return Identity()
        if scaling_type == 'sigmoid':
            return Sigmoid()
        else:
            print("Distance type does not exist")


# ----------------------------------------------------------------------------------------------------------------------

