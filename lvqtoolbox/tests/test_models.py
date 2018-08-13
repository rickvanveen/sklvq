import scipy as sp
import numpy as np
from sklearn import datasets
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing

from lvqtoolbox.glvq import _relative_distance_difference_cost, _squared_euclidean

from scipy.optimize import minimize


def test_costfun():
    iris = datasets.load_iris()

    # classifier = GLVQClassifier()
    # classifier = classifier.fit(iris.data, iris.target)
    #
    # accuracy = classifier.score(iris.data, iris.target)
    #
    # #TODO: Create some more tests...

    iris.data = preprocessing.scale(iris.data)

    prototype_labels = np.transpose(unique_labels(iris.target))

    prototypes = np.zeros((prototype_labels.size, iris.data.shape[1]))
    for ilabel in range(prototype_labels.size): # TODO: Move to common and probably can be improved with some clever list comprehension stuff
        prototypes[ilabel, :] = np.mean(iris.data[prototype_labels[ilabel] == iris.target, :], axis=0)

    initial_prototypes = prototypes.reshape([1, prototype_labels.size * iris.data.shape[1]])
    optimised_prototypes = minimize(_relative_distance_difference_cost,
                                    initial_prototypes,
                                    (prototype_labels, iris.data, iris.target, _squared_euclidean),
                                    'L-BFGS-B',
                                    options={'maxiter': 1000, 'disp': True},
                                    tol = 1e-6)
    final_prototypes = optimised_prototypes.x.reshape([prototype_labels.size, iris.data.shape[1]])
    print(optimised_prototypes.x)
    print("\n\n")
    print(final_prototypes)
    print("\n\n")
    print(optimised_prototypes.fun)


