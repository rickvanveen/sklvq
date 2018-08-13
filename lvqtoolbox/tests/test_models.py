import scipy as sp
import numpy as np
from sklearn import datasets
from sklearn.utils.multiclass import unique_labels

from lvqtoolbox import GLVQClassifier


def test_costfun():
    iris = datasets.load_iris()

    classifier = GLVQClassifier()
    classifier = classifier.fit(iris.data, iris.target)

    accuracy = classifier.score(iris.data, iris.target)

    #TODO: Create some more tests...


    # prototype_labels = unique_labels(iris.target)
    #
    # prototypes = np.zeros((prototype_labels.size, iris.data.shape[1]))
    # for ilabel in range(prototype_labels.size): # TODO: Move to common and probably can be improved with some clever list comprehension stuff
    #     prototypes[ilabel, :] = np.mean(iris.data[prototype_labels[ilabel] == iris.target, :], axis=0)
    #
    # initial_prototypes = prototypes.reshape([1, prototype_labels.size * iris.data.shape[1]])
    # optimised_prototypes = fmin_cg(std_costfun, initial_prototypes, args=(prototype_labels, iris.data, iris.target, squared_euclidean))
    # final_prototypes = optimised_prototypes.reshape([prototype_labels.size, iris.data.shape[1]])
    pass
