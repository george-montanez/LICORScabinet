from __future__ import division
import numpy as np
from scipy.stats.kde import gaussian_kde
from scipy.spatial.distance import cdist
from multi_flatten import multi_flatten

class VectorGaussianKernel(object):
    def KDE_evaluation(self, distances, bandwidths, dim, num_pts, weights=None):
        if weights is None:
            weights = np.ones(distances.shape)
        consts_1 = 1./(bandwidths**dim * np.sqrt(2*np.pi))
        consts_2 = np.expand_dims(-0.5/(bandwidths**2), axis=1)
        return 1./num_pts * consts_1 * (np.exp(consts_2 * np.power(distances, 2)) * weights).sum(axis=1)    

    def rule_of_thumb_bandwidth(self, data, n):
        """ Rule-of-Thumb """
        data = multi_flatten(data)
        d = data.shape[1]
        a = 1. / (d + 4)
        """ Scalar Covariance """
        stds = np.std(data, axis=0)
        return ((4./(d+2))**a) * (np.mean(stds).item()) * (n**-a)

    def compute_cv_bandwidth(self, sample):
        from sklearn.grid_search import GridSearchCV
        from sklearn.neighbors import KernelDensity
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv=20)
        grid.fit(sample)
        return grid.best_params_['bandwidth']


if __name__ == "__main__":
    N = 10000
    distances = np.random.random((N,10))
    bandwidths = np.random.random(N)
    VGK = VectorGaussianKernel()
    print VGK.KDE_evaluation(distances, bandwidths, dim=9, N=1000000).shape



