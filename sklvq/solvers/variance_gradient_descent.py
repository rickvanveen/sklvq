# Stochastic gradient descent
# vSGD maintains three exponential moving averages: of the gradient, g, of the Hadamard squared gradient, v,
# and of the Hesse diagonal, h
# Only hyperparameter C > 1, which does something that has to do with initialization.
import numpy as np

from objectives import ObjectiveBaseClass
from sklvq.solvers.base import SolverBaseClass

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sklvq.models import LVQClassifier


class VarianceGradientDescent(SolverBaseClass):

    def __init__(self, c=0.5):
        self.c = c

    def solve(self, data: np.ndarray, labels: np.ndarray, objective: ObjectiveBaseClass,
              model: 'LVQClassifier') -> 'LVQClassifier':

