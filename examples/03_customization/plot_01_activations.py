"""
====================
Activation Functions
====================
"""

from sklearn.datasets import load_iris
from sklearn.metrics import classification_report

from sklvq import GLVQ

data, labels = load_iris(return_X_y=True)

###############################################################################
# The sklvq contains already a few activation function. Please see the API reference
# under Documentation. However, it is fairly easy to create your own. The package
# works with callable classes and provides a base class for convenience. The base class
# for the activation functions is sklvq.activations.ActivationBaseClass` and does
# nothing more then tell you to implement a `__call__()` and `gradient()` method.

import numpy as np
from typing import Union

from sklvq.activations import ActivationBaseClass


# This is the implementation of sklvq.activations.Sigmoid with some additional comments
class CustomSigmoid(ActivationBaseClass):

    # Activation callables can have a custom init of which the parameters can be passed
    # through the `activation_params (Dict)' parameter of the LVQ algorithms. Or the
    # object can just be initialized before hand.
    def __init__(self, beta: Union[int, float] = 1):
        self.beta = beta

    # The activation call function needs to apply the activation elementwise on x.
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(1 / (np.exp(-self.beta * x) + 1))

    # The gradient is the elementwise derivative of the activation function.
    def gradient(self, x: np.ndarray) -> np.ndarray:
        exp = np.exp(self.beta * x)
        return np.asarray((self.beta * exp) / (exp + 1) ** 2)


###############################################################################
# The CustomSigmoid above, accompanied with some tests and documentation, would make a
# great addition to the sklvq package. However, it can also directly be passed to
# the algorithm.

model = GLVQ(activation_type=CustomSigmoid, activation_params={"beta": 2})

model.fit(data, labels)

# Predict the labels using the trained model
predicted_labels = model.predict(data)

# Print a classification report (sklearn)
print(classification_report(labels, predicted_labels))
