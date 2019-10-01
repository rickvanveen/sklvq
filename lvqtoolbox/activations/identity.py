from . import ActivationBaseClass


class Identity(ActivationBaseClass):

    def __int__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        """ Implements the identity function: f(x) = x

        Note helps with single interface in cost function.

        Parameters
        ----------
        x : ndarray

        Returns
        -------
        x : ndarray, same shape and values as input.
        """
        return x

    def gradient(self, x):
        """ Implements the identity function derivative: g(x) = 1

        Note helps with single interface in cost function.

        Parameters
        ----------
        x : Anything

        Returns
        -------
        gradient : scalar,
                   Returns the constant 1 no matter the shape or values of the input.
        """
        return 1
