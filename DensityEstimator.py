from __future__ import division
import numpy as np
from multi_flatten import multi_flatten

class DensityEstimator(object):
    def __init__(self, data_points, num_subsamples=-1, num_pts_used=-1, **kwargs):
        N = data_points.shape[0]
        self.num_pts_used = num_pts_used
        if num_subsamples != -1:
            num_pts = min(num_subsamples, N)
            self.sampled_indices = np.random.choice(N, size=num_pts, replace=False)
            self.points = data_points[self.sampled_indices]
        else:
            self.points = data_points
            self.sampled_indices = range(N)

    def __call__(self, query_points, **kwargs):
        return np.ones(query_points.shape[0]) * -1

