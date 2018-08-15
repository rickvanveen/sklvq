import numpy as np


# f(x) = 1 / (e^{-Bx} + 1)
def sigmoid(x, beta=2):
    return 1 / (np.exp(-beta * x) + 1)


def sigmoid_grad(x, beta=2):
    return (beta * np.exp(beta*x)) / (np.exp(beta*x) + 1)**2


# f(x) = x
def identity(x, **kwargs):
    return x


def identity_grad(*args, **kwargs):
    return 1
