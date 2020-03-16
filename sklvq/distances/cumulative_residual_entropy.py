import numpy as np
import scipy as sp

from statsmodels.distributions.empirical_distribution import ECDF

from .base import DistanceBaseClass

# Implementation of cre to be used as distance measure in GLVQ (not GMLVQ) or we need to define a new cre
# including the idea of this adaptive matrix.


class CumulativeResidualEntropy(DistanceBaseClass):

    def gradient(self, data: np.ndarray, model: 'LVQClassifier', i_prototype: int) -> np.ndarray:
        pass

    def __init__(self, n_bins=None):
        self.n_bins = n_bins

    def __call__(self, data: np.ndarray, model: 'LVQClassifier') -> np.ndarray:
        """
            Parameters
            ----------
            data       : ndarray, shape = [n_obervations, n_features]
                        Inputs are converted to float type.
            model : ndarray, shape = [n_prototypes, n_features]
                         Inputs are converted to float type.

            Returns
            -------
            distances : ndarray, shape = [n_observations, n_prototypes]
                        The dist(u=XA[i], v=XB[j]) is computed and stored in the
                        ij-th entry.
        """
        prototypes = model.prototypes_

        ccre = np.zeros((data.shape[0], prototypes.shape[0]))

        if self.n_bins is None: #
            self.n_bins = np.floor(model.prototypes_.shape[1] / 10)

        max_data = np.max(data)
        min_data = np.min(data)
        n_steps = data.size + 1
        step_size = (max_data - min_data) / n_steps
        lambdas = np.linspace(min_data, max_data + step_size, n_steps)

        # lambdas = np.linspace(np.min(data), np.max(data), data.size)

        # define the bins for the prototype(s) Equally spaced bins
        p_bins = np.linspace(np.min(prototypes), np.max(prototypes), prototypes.size)
        for i, sample in enumerate(data):
            sample_cre = _cre(sample, lambdas)
            # Loop over prototype
            for j, prototype in enumerate(prototypes):
                # Loop over bins
                sw_cre = 0
                for k in range(0, p_bins.size - 1):
                    # Get subset of sample for which prototype is within current bin

                    # check if prototypes has data in this bin...
                    if not np.any(((prototype >= p_bins[k]) & (prototype < p_bins[k + 1]))):
                        continue

                    subset = np.array(sample[((sample >= p_bins[k]) & (sample < p_bins[k + 1]))])
                    # Check if sample has data in this bin if yes, compute cre if not 0

                    if subset.size == 0:
                        continue

                    # Calculate CRE for this subset
                    cre = _cre(subset, lambdas)

                    # Calculate weight proportional to size(subset)/size(sample)
                    weight = subset.size / sample.size

                    # Calculate weigth * CRE(subset)
                    wcre = weight * cre

                    # Sum all values across subsets. ==> expectation value of cre given prototype
                    sw_cre += wcre
                # calculate CCRE=cre-E(cre|PT)
                ccre[i, j] = sample_cre - sw_cre

        return ccre


# def _ccre(data: np.ndarray, prototypes: np.ndarray, lambdas:np.ndarray) -> np.ndarray:
#     pass


def _cre(datum: np.ndarray, lambdas:np.ndarray) -> float:
    # if datum.size == 1:
    #     return 0.0
    ecdf = ECDF(datum, side='left')

    fc = 1 - ecdf(lambdas)

    with np.errstate(divide='ignore', invalid='ignore'):
        log_fc = np.log(fc)

    # Log undefined for 0
    log_fc[fc == 0] = 0

    # Hint: FC == 0 gaat fout misschien als er maar 1 element in datum zit (datum kan dus een subset zijn waardoor dat voorkomt)
    cre = -1 * np.sum(fc * log_fc) / np.sum(fc)

    return cre


