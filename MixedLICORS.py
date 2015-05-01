from __future__ import division
from Queue import PriorityQueue
import math
import random
import numpy as np
import scipy
from scipy.cluster.vq import kmeans2
from scipy import sparse
import LightConeExtractor as LCE
from WeightedKDE import WeightedKDE as DenEst
from time import time
from multi_flatten import multi_flatten

FUDGE_FACTOR = 1e-30
MAX_ITERATIONS = 100

class MixedLICORS(object):
    def __init__(self, K_max, delta=1., verbose=False, fold_number=1, 
            track_weights=False, fixed_bandwidth=True, kmeans_init=True,
            n_subsamples=-1, cache=True, fully_nonparametric=False):
        """ Load with placeholder variables. Will replace
            once we load with data. """
        self.K_max = K_max            
        self.W = self.random_W_initialization(1, K_max)
        self.light_cones = None
        self.N = lambda : self.W.shape[0]
        self.K = lambda : self.W.shape[1]
        self.current_estimates = {                     
                                  "WEIGHTED_PLC_MEANS" : {}, 
                                  "WEIGHTED_COV_MATRIX" : {},
                                  "COMPONENT_MEANS" : {},
                                 }
        self.verbose = verbose
        self.delta = delta
        self.states = []
        self.PLC_KDE = None
        self.FLC_KDE = None
        self.fold_number = fold_number
        self.track_weights = track_weights
        self.fixed_bandwidth = fixed_bandwidth
        self.kmeans_init = kmeans_init
        self.default_covariance_matrix = None
        self.subsample_limit = int(n_subsamples)
        self.fully_nonparametric = fully_nonparametric

    def random_W_initialization(self, N, K):
        W = np.zeros((N, K)).astype("float32")
        assignments = np.random.randint(K, size=(N))
        for row, col in enumerate(assignments):
            W[row, col] += 1.
        """ Make sure each state has at least one light cone """                
        for j in range(K):
            if not W[:,j].sum():
                row_index = np.random.randint(N)
                W[row_index] = 0
                W[row_index, j] = 1.
        return W

    def bad_W_initialization(self, N, K):
        return np.random.randint(2, size=(N, K)).astype("float32")

    def kmeanspp_initialization(self, N, K):
        centroids, assignments = kmeans2(self.get_FLCs().reshape((-1,1)), K)
        W = np.zeros((N, K)).astype("float32")
        for row, col in enumerate(assignments):
            W[row, col] += 1.
        return W

    def create_initialized_W_matrix(self, N, K):
        if self.kmeans_init:
            return self.kmeanspp_initialization(N, K)
        else:
            return self.random_W_initialization(N, K)

    def import_model(self, W):
        self.W = W
        self.N = lambda : self.W.shape[0]
        self.K = lambda : self.W.shape[1]
        self.refresh_current_estimates()

    def load(self, data):
        """ Takes data matrix and parses it into 
            PLCs. Uses LightConeExtractor class to extract 
            PLC/FLC arrays from data matrix.
            NOTE: Light cones have the FLC (current space-time
            point) as the last (multidimensional) point of each row.
        """
        lce = LCE.LightConeExtractor()
        light_cones = lce.extract(data)
        self.load_light_cones(light_cones)

    def load_light_cones(self, light_cones):
        """ Split into training and test/hold out sets """
        total = len(light_cones)
        indices = np.random.permutation(total)
        self.light_cones = light_cones[indices[:int(total * .75)]]
        self.test_light_cones = light_cones[indices[int(total * .25):]]
        if not len(self.light_cones):
            raise Error("Must have at least one light cone.")
        PLCs = self.get_PLCs()
        FLCs = self.get_FLCs()
        self.default_covariance_matrix = np.cov(multi_flatten(PLCs), rowvar=0)
        self.PLC_KDE = DenEst(d_points=PLCs, num_subsamples=500, fixed_bandwidth=self.fixed_bandwidth, mode="FULL")
        self.PLC_KDE.update_bandwidth(self.PLC_KDE.calculate_bandwidth(PLCs))
        self.FLC_KDE = DenEst(d_points=FLCs, num_subsamples=500, fixed_bandwidth=self.fixed_bandwidth, mode="FULL")
        self.FLC_KDE.update_bandwidth(self.FLC_KDE.calculate_bandwidth(FLCs))

    def get_FLCs(self, light_cones=None):
        if not light_cones is None:
            return np.array(light_cones)[:,-1]
        else:
            return self.light_cones[:,-1]

    def get_PLCs(self, light_cones=None):
        if not light_cones is None:
            return np.array(light_cones)[:,:-1]
        else:
            return self.light_cones[:,:-1]   

    def refresh_current_estimates(self):
        self.current_estimates = {  
                                  "WEIGHTED_PLC_MEANS" : {}, 
                                  "WEIGHTED_COV_MATRIX" : {},
                                  "COMPONENT_MEANS" : {},
                                 }
        for j in range(self.W.shape[1]):
            weights = self.W[:,j].copy() + FUDGE_FACTOR
            normalized_weights = weights / weights.sum()
            self.current_estimates["COMPONENT_MEANS"][j] = np.dot(normalized_weights, self.get_FLCs())
            self.current_estimates["WEIGHTED_PLC_MEANS"][j] = self.weighted_mean(j)
            self.current_estimates["WEIGHTED_COV_MATRIX"][j] = self.weighted_covariance_matrix(j)

    def is_pos_semi_def(self, S):
        return np.all(np.linalg.eigvals(S) >= 0)

    def weighted_mean(self, j):
        weights = self.W[:,j].copy() + FUDGE_FACTOR
        normalized_weights = weights / weights.sum()
        PLCs = self.get_PLCs() 
        return np.average(PLCs, 
                          axis=0,
                          weights=normalized_weights
                         ).ravel()

    def weighted_covariance_matrix(self, j):
        weights = self.W[:,j].copy()
        while np.isclose(weights.sum(), 1.):
            weights += 1 / len(weights)
        weight_sum = weights.sum()
        sum_squared_weights = np.sum(weights**2)
        if weight_sum**2 == sum_squared_weights:
            print "ERROR"
            print weight_sum**2, sum_squared_weights
        normalizing_const = weight_sum / (weight_sum**2 - sum_squared_weights)
        weights_arr = weights.reshape((-1,1))
        PLCs = self.get_PLCs()
        plcs_demeaned = PLCs - PLCs.mean(axis=0)
        plcs_demeaned = plcs_demeaned.reshape((plcs_demeaned.shape[0], -1))
        A = np.matrix(weights_arr * np.array(plcs_demeaned)).T
        B = plcs_demeaned
        result = (normalizing_const * np.dot(A,B))
        if not self.is_pos_semi_def(result):
            return self.default_covariance_matrix
        else:
            return result

    def f_hat_conditional_densities(self, query_points, label=None):
        """ Uses weighted KDE to estimate f(x_i | S_i = s_j)
            (eq. 21 of MixedLICORS paper), for each extremal 
            state j.
            Returns matrix of real-valued density estimates,
            rows index each point, columns are for each component state.
        """
        results = []
        FLCs = self.get_FLCs()
        for j in range(self.K()):
            if not self.fixed_bandwidth:
                argmax_data = FLCs[np.array(self.W[:,j] == np.max(self.W,axis=1)).ravel()]
                if len(argmax_data) > 1:
                    self.FLC_KDE.update_bandwidth(self.FLC_KDE.calculate_bandwidth(argmax_data))
            weights = self.W[:,j].copy()
            res = self.FLC_KDE(query_points, weights, label=label)
            results.append(res.ravel())
        return np.matrix(results).T

    def PLC_densities(self, j, query_points, label=None, do_not_cache=False):
        if self.fully_nonparametric:
            return self.PLC_densities_KDE(j, query_points, label=label, do_not_cache=do_not_cache)
        else:
            return self.PLC_densities_gaussian(j, query_points)

    def PLC_densities_gaussian(self, j, query_points):
        query_points = multi_flatten(query_points)
        m = self.current_estimates["WEIGHTED_PLC_MEANS"][j]
        c = self.current_estimates["WEIGHTED_COV_MATRIX"][j] 
        return scipy.stats.multivariate_normal.pdf(query_points, m, c)

    def PLC_densities_KDE(self, j, query_points, label=None, do_not_cache=False):
        PLCs = self.get_PLCs()
        if not self.fixed_bandwidth:
            argmax_data = PLCs[np.array(self.W[:,j] == np.max(self.W,axis=1)).ravel()]
            if len(argmax_data) > 1:
                self.PLC_KDE.update_bandwidth(self.PLC_KDE.calculate_bandwidth(argmax_data))
        weights = self.W[:,j].copy()
        return self.PLC_KDE(query_points, weights, label=label, do_not_cache=do_not_cache)

    def update_weights(self):
        """ Updates all weights in W weight matrix using
            current estimates.
        """
        if self.verbose: print "Refreshing current estimates..."
        self.refresh_current_estimates()
        f_hat_matrix = self.f_hat_conditional_densities(self.get_FLCs(), label="TRAIN_FLCs")
        PLCs = self.get_PLCs()
        for j in xrange(self.K()):
            if self.verbose: print "updating state %d" % (j,)
            density_array = np.expand_dims(self.PLC_densities(j, PLCs, label="UPDATE_WEIGHTS"), axis=1)
            n_hat_ratio = self.W[:,j].sum() / self.N()
            result = np.array(f_hat_matrix[:,j]) * density_array * n_hat_ratio
            self.W[:,j] = result.ravel()
        self.renormalize_weights()
        self.refresh_current_estimates()

    def renormalize_weights(self):
        self.W += FUDGE_FACTOR      
        self.W = self.W / np.sum(self.W, axis=1).reshape((-1,1))
        if not np.all(np.isclose(self.W.sum(axis=1), 1.)):
            print "RENORMALIZATION ERROR"
            print self.W
            print self.W.sum(axis=1)            

    def remove_small_weight_states(self):
        K = self.K()
        strong_states = (np.sum(self.W, axis=0) > 1.5)
        weak_states = np.logical_not(strong_states)
        self.W = self.W[:,strong_states]
        if self.verbose and np.any(weak_states): 
            print "Removing states", np.arange(len(weak_states))[weak_states]
        self.renormalize_weights()
        self.refresh_current_estimates()

    def predict(self, PLC):
        return self.predict_batch(PLC)

    def predict_batch(self, points, points_label=None, do_not_cache=False):
        K = self.K()
        N = self.N()    
        lightcone_densities = np.vstack([self.PLC_densities(j, points, label=points_label, do_not_cache=do_not_cache) for j in range(K)]).T
        n_hats = self.W.sum(axis=0) / N
        raw_component_weights = lightcone_densities * n_hats + FUDGE_FACTOR        
        normalized_component_weights = raw_component_weights / np.expand_dims(np.sum(raw_component_weights, axis=1), axis=1)
        component_means = np.array([self.current_estimates["COMPONENT_MEANS"][j] for j in range(K)])   
        return np.expand_dims(np.dot(normalized_component_weights, component_means), axis=1)

    def predict_states(self, points):
        K = self.K()
        N = self.N()
        lightcone_densities = np.vstack([self.PLC_densities(j, points, label="PREDICT_STATES") for j in range(K)]).T
        n_hats = self.W.sum(axis=0) / N
        raw_component_weights = lightcone_densities * n_hats + FUDGE_FACTOR        
        normalized_component_weights = raw_component_weights / np.expand_dims(np.sum(raw_component_weights, axis=1), axis=1)        
        state_assignments = np.argmax(normalized_component_weights, axis=1)       
        return np.array([self.current_estimates["COMPONENT_MEANS"][i] for i in state_assignments])

    def merge_closest_states(self):
        results_matrix = self.f_hat_conditional_densities(self.get_FLCs(), label="ALL_FLCs")
        smallest_distance = (float("inf"), None)
        for i in range(self.K()):
            for j in range(i + 1, self.K()):
                d = np.abs(results_matrix[:,i] - results_matrix[:,j]).mean()
                if d < smallest_distance[0]:
                    smallest_distance = (d, (i, j))
        a, b = smallest_distance[1]
        if self.verbose: print "Merging states %d and %d" % (a,b)
        self.W[:,a] += self.W[:,b]
        self.W = np.delete(self.W, b, axis=1)        

    def compute_likelihoods(self, PLCs, FLCs):
        K = self.K()
        N = self.N()
        future_given_state_probs = np.nextafter(self.f_hat_conditional_densities(FLCs, label="PDF_FLCs"), 1.)
        state_given_past_probs = np.nextafter(np.vstack([self.PLC_densities(j, PLCs) for j in range(K)]), 1.).T        
        ''' Weight by state likelihood '''
        n_hats = self.W.sum(axis=0) / N
        state_given_past_probs *= n_hats
        state_given_past_probs = np.nextafter(state_given_past_probs, 1.)
        ''' Normalize '''
        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)
        ''' Return mixed likelihoods '''
        return np.nextafter(np.sum(np.multiply(state_given_past_probs, future_given_state_probs), axis=1), 1.)

    def log_likelihood(self, light_cone_seq):
        """ Calculates log-likelihood for a sequence of light_cones. Useful for classification.            
        """
        PLCs = self.get_PLCs(light_cones=light_cone_seq)
        FLCs = self.get_FLCs(light_cones=light_cone_seq)       
        return np.log(self.compute_likelihoods(PLCs, FLCs)).sum(axis=0)

    def get_current_MSE(self):
        test_PLCs = self.get_PLCs(light_cones=self.test_light_cones)
        test_FLCs = self.get_FLCs(light_cones=self.test_light_cones).reshape((-1, 1))
        preds = self.predict_batch(test_PLCs, points_label="TRAINING_TEST_PLCs").reshape((-1, 1))
        return ((preds - test_FLCs)**2).mean(axis=0)

    def save_weights(self, num_states, iteration):
        np.savetxt('../Results/MXL/weights_%d_%d_%d_%d.txt' % (self.N(), self.fold_number, num_states, iteration), self.W)

    def learn(self, histories, futures):
        """ Main loop of Mixed LICORS. 
        """
        delta = self.delta
        best_W_tuple = None        
        light_cones = np.hstack((histories, futures))
        """ Sub-sample a smaller set of light cones for tractability """
        if self.subsample_limit > 0:
            limit = min(self.subsample_limit, light_cones.shape[0])
            np.random.shuffle(light_cones)
            light_cones = light_cones[:limit]
        """ Load light cones """
        self.load_light_cones(light_cones)
        """ Initialize Weight Matrix """
        N = self.light_cones.shape[0]
        self.W = self.create_initialized_W_matrix(N, self.K_max)
        """ Iterate """
        while self.K() > 0:
            converged = False
            iteration = 0
            """ Run until convergence or max iterations reached. """
            while not converged and iteration < MAX_ITERATIONS:
                self.remove_small_weight_states() 
                old_W = self.W.copy()
                self.update_weights()                
                if self.track_weights: self.save_weights(self.K(), iteration)
                MSE = self.get_current_MSE()
                if (not best_W_tuple) or MSE < best_W_tuple[0]:
                    if self.verbose: print "New Best MSE", MSE, self.K()
                    best_W_tuple = (MSE, self.W.copy())                    
                difference = np.max(np.sqrt(np.mean((self.W - old_W)**2, axis=1)))
                converged = (difference < delta)
                if self.verbose: 
                    print "Normed difference", difference, delta, converged, self.K(), best_W_tuple[0], MSE
                iteration += 1
            """ Merge step. """                 
            if self.K() == 1:
                break            
            else:
                self.merge_closest_states()                          
        """ Save W^* with lowest out-of-sample MSE """
        self.import_model(best_W_tuple[1])

    def save_params(self, fold):    
        np.savetxt('../Results/MXL/best_model_%d_%d_%d.txt' % (self.N(), self.K_max, fold), self.W)
        f = open('../Results/MXL/best_k_%d_%d_%d.txt' % (self.N(), self.K_max, fold), 'w')
        f.write(str(self.K()))

    def print_state_info(self):
        for j in range(self.K()):
            print j, self.current_estimates["COMPONENT_MEANS"][j], self.W[:,j].sum()


def main():
    sample_len = 10000
    states = 2
    h_p = 3
    d = 1
    pml = MixedLICORS(states, True)
    light_cones = np.random.random((sample_len,h_p,d))
    light_cones_2 = np.random.random((10,h_p,d)) * 2.25
    pml.load_light_cones(light_cones)
    pml.learn(light_cones[:,:-1], light_cones[:,-1:])
    print pml.log_likelihood(light_cones[:10]).item()
    print pml.log_likelihood(light_cones_2[:10]).item()

if __name__ == "__main__":
    main()
