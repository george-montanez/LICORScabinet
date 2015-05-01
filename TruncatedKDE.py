from __future__ import division
import numpy as np
from sklearn.neighbors import NearestNeighbors
from DensityEstimator import DensityEstimator
from VectorGaussianKernel import VectorGaussianKernel
from multi_flatten import multi_flatten

class TruncatedKDE(DensityEstimator):
    def __init__(self, d_points, num_subsamples=-1, num_pts_used=-1, d=None, **kwargs):
        super(TruncatedKDE, self).__init__(d_points, num_subsamples, num_pts_used)
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
        self.VGK = VectorGaussianKernel()
        self.bw = self.VGK.rule_of_thumb_bandwidth(self.points, num_pts)
        self.nbrs = NearestNeighbors(n_neighbors=num_pts, algorithm='ball_tree').fit(self.points)

    def __call__(self, query_points, **kwargs):
        return self.kde(query_points)

    def kde(self, query_points):
        pts = multi_flatten(query_points)
        bandwidths = np.ones(pts.shape[0]) * self.bw
        distances, indices = self.nbrs.kneighbors(pts)
        num_pts = min(self.num_of_neighbors, self.points.shape[0])        
        return self.VGK.KDE_evaluation(distances, bandwidths, self.dim, num_pts)

