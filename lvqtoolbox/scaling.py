import numpy as np


# TODO: Potentially add *args and **kwargs to ignore other input...
# f(x) = 1 / (e^{-Bx} + 1)
def sigmoid(x, beta=2):
    return 1 / (np.exp(-beta * x) + 1)


# TODO: Potentially add *args and **kwargs to ignore other input...
def sigmoid_grad(x, beta=2):
    return (beta * np.exp(beta*x)) / (np.exp(beta*x) + 1)**2


# f(x) = x
def identity(x, **kwargs):
    return x


def identity_grad(*args, **kwargs):
    return 1
