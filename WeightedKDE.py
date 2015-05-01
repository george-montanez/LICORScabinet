from __future__ import division
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from DensityEstimator import DensityEstimator
from VectorGaussianKernel import VectorGaussianKernel
from multi_flatten import multi_flatten

class WeightedKDE(DensityEstimator):
    def __init__(self, d_points, num_subsamples=-1, num_pts_used=-1, d=None, kernel='gaussian', h=None, fixed_bandwidth=False, mode="FULL"):
        super(WeightedKDE, self).__init__(d_points, num_subsamples, num_pts_used)
        if not d:
            if len(self.points.shape) < 2:
                d = 1
            else:                
                d = self.points.shape[1]
        self.kernel = kernel
        self.mode = mode
        self.h = h
        self.dim = d
        if not self.h:
            self.update_bandwidth(self.calculate_bandwidth(self.points))
        else:            
            self._update_norm_consts()
        self.xs = self.points        
        self.kde = self.good_kde        
        self.saved_results = {}
        self.fixed_bandwidth = fixed_bandwidth

    def __call__(self, query_points, weights=None, label=None, do_not_cache=False):
        return self.kde(query_points, weights, label, do_not_cache)

    def _update_norm_consts(self):
        h = self.h
        d = self.dim
        if not h is None:
            if self.mode == "SCALAR":
                self._norm_const_1 = 1./(h**d * np.sqrt(2*np.pi))
            else:
                self._norm_const_1 = 1./(np.linalg.det(h) * (2*np.pi)**(d/2.))

    def get_norm_consts(self):
        return self._norm_const_1

    def update_bandwidth(self, h):
        self.h = h
        self._update_norm_consts()

    def chunk_process(self, chunk1, chunk2):
        if self.mode == "SCALAR":
            c2 = -0.5/self.h**2            
            distances = cdist(chunk1, chunk2, 'euclidean')**2
            return np.exp(c2 * distances)
        else:            
            HI = np.linalg.inv(self.h)
            distances = cdist(chunk1, chunk2, 'mahalanobis', VI=HI**2)
            return np.exp(-0.5 * distances)        
    
    def inner_loop(self, const1, weights, chunk, state_points, block_length):
        M = state_points.shape[0]
        j_results = []
        start_j = 0
        end_j = 0
        for block_j in range(M // block_length):
            start_j = block_j * block_length
            end_j = start_j + block_length
            cp = self.chunk_process(chunk, state_points[start_j:end_j])
            j_results.append(cp)        
        """ compute remaining distances for j block """
        cp = self.chunk_process(chunk, state_points[end_j:])
        j_results.append(cp)
        stacked = np.hstack(j_results)
        return const1 * np.dot(stacked, weights), stacked

    def good_kde(self, query_points, weights, label, do_not_cache):
        xs = multi_flatten(self.xs)
        query_points = multi_flatten(query_points)
        N = query_points.shape[0]
        M = xs.shape[0]
        if self.h is None:
            self.update_bandwidth(self.calculate_bandwidth(xs))
        if weights is None:
            weights = np.ones(M)
        else:
            weights = weights[self.sampled_indices]
        const1 = self._norm_const_1
        if self.fixed_bandwidth and label in self.saved_results:
            stacks = self.saved_results[label]
            all_results = [const1 * np.dot(s, weights) for s in stacks]
            return 1./weights.sum() * np.hstack(all_results)
        else:
            block_length = 500
            all_results = []
            d_stacks = []
            stacks = []
            start_i = 0
            end_i = 0
            for block_i in range(N // block_length):
                start_i = block_i * block_length
                end_i = start_i + block_length
                inner_result, stacked = self.inner_loop(const1, weights, query_points[start_i:end_i], xs, block_length)                
                all_results.append(inner_result)
                if self.fixed_bandwidth:
                    stacks.append(stacked)
            """ Compute for remaining block """
            inner_result, stacked = self.inner_loop(const1, weights, query_points[end_i:], xs, block_length)
            all_results.append(inner_result)
            if self.fixed_bandwidth and not do_not_cache:
                stacks.append(stacked)
                self.saved_results[label] = stacks
            return 1./weights.sum() * np.hstack(all_results)

    def is_pos_semi_def(self, S):
        return np.all(np.linalg.eigvals(S) >= 0)

    def calculate_bandwidth(self, data):
        """ Rule-of-Thumb """
        data = multi_flatten(data)
        n = data.shape[0]
        d = data.shape[1]
        a = 1. / (d + 4)
        if self.mode == "FULL":
            """ Full Covariance """            
            if d == 1:
                s_mat = np.std(data, axis=0) * np.eye(d)
            else:
                cov = np.cov(data, rowvar=0)
                if self.is_pos_semi_def(cov):
                    s_mat = scipy.linalg.sqrtm(cov)
                else:
                    s_mat = np.std(data, axis=0) * np.eye(d)              
            return ((4./(d+2))**a) * s_mat * (n**-a)
        elif self.mode in ("DIAG", "DIAGONAL"):
            """ Diagonal Covariance """
            cov_mat = np.std(data, axis=0) * np.eye(d)
            return ((4./(d+2))**a) * cov_mat * (n**-a)
        else: 
            """ Scalar Covariance """
            stds = np.std(data, axis=0)
            return ((4./(d+2))**a) * (np.mean(stds).item()) * (n**-a)

