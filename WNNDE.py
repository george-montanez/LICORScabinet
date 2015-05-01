from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import special
from DensityEstimator import DensityEstimator
from multi_flatten import multi_flatten

class WNNDE(DensityEstimator):
    def __init__(self, d_points, num_subsamples=-1, num_pts_used=-1, d=None, **kwargs):
        super(WNNDE, self).__init__(d_points, num_subsamples, num_pts_used)
        if num_pts_used == -1:
            num_pts_used = self.points.shape[0]
        self.num_of_neighbors = num_pts_used
        if not d:
            if len(self.points.shape) < 2:
                d = 1
            else:                
                d = self.points.shape[1]
        self.dim = d
        num_pts = min(self.num_of_neighbors, self.points.shape[0])
        self.nbrs = NearestNeighbors(n_neighbors=num_pts, algorithm='ball_tree').fit(self.points)

    def __call__(self, query_points, weights = None, **kwargs):
        return self.pdf(query_points, weights)

    def pdf(self, query_points, weights):
        M = self.points.shape[0]
        if weights is None:
            weights = np.ones(M)
        else:
            weights = weights[self.sampled_indices]
        pts = multi_flatten(query_points)
        distances, indices = self.nbrs.kneighbors(pts)
        eps = distances[:, -1]
        d = self.dim
        f = np.take(weights, indices).sum(axis=1) / M * special.gamma(d/2. + 1) / (np.pi**(d/2.) * eps**d)
        return f
