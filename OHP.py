from __future__ import division
import numpy as np
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from time import time
from multi_flatten import multi_flatten
from cluster import *
from WeightedKDE import WeightedKDE as DenEst

class OHP(object):
    class predictive_state(object):        
        def __init__(self, histories, futures, N):
            ''' Assumes histories and futures already flattened '''
            self.num_of_kde_points = 500
            self.mean_future = np.mean(futures, axis=0)
            self.mean_history = np.mean(histories, axis=0)
            self.total_points = N
            self.state_points = histories.shape[0]
            self.FLC_DE = DenEst(futures, num_subsamples=self.num_of_kde_points, mode='FULL')
            self.PLC_DE = DenEst(histories, num_subsamples=self.num_of_kde_points, mode='FULL')            

        def distance_to_state(self, point):
            point = point.reshape((1,-1))
            return cdist(point, self.mean_history, 'euclidean').item()
           
        def batch_lc_likelihood_given_state(self, points):
            return self.PLC_DE(multi_flatten(points))

        def batch_emission_likelihood_given_state(self, futures):
            return self.FLC_DE(multi_flatten(futures))

        def state_likelihood(self):
            return self.state_points / self.total_points

    def __init__(self, K_max, cluster_epsilon=0.5, fold_number=1, verbose=True):
        ''' Initialization '''
        self.K_max = K_max
        self.state_weights = []
        self.states = []
        self.cluster_assignments = []
        self.cluster_epsilon = cluster_epsilon
        self.fold_number = fold_number
        self.verbose = verbose

    def learn(self, histories, futures):        
        ''' Flatten out multidimensional X into 2D array'''
        flat_histories = multi_flatten(histories)
        flat_futures = multi_flatten(futures)
        N = flat_histories.shape[0]
        ''' Cluster by futures'''
        mbkm = MiniBatchKMeans(n_clusters=self.K_max, init='k-means++', compute_labels=True)      
        mbkm.fit(flat_futures)
        labels = mbkm.labels_
        self.cluster_assignments = np.ones(len(flat_histories)) * -1
        ''' Create states from cluster assignment '''
        for label in set(labels):
            vectors_in_state = np.equal(labels, label)
            if vectors_in_state.sum() > 1:
                ps = self.predictive_state(flat_histories[vectors_in_state], flat_futures[vectors_in_state], N)
                self.cluster_assignments[vectors_in_state] = label
                self.states.append(ps)        
        ''' Create mean futures and histories for states '''            
        self.state_mean_futures_array = np.array([s.mean_future for s in self.states])
        self.state_mean_histories_array = np.array([s.mean_history for s in self.states])    

    def predict_batch_nearest(self, pasts, points_label=None, do_not_cache=False):
        ''' Uses most likely state prediction '''
        state_given_past_probs = []
        for k, s in enumerate(self.states):
            if self.verbose: print "Evaluating state", k
            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)
        ''' Weight by state likelihood '''
        likelihoods = np.array([s.state_likelihood() for s in self.states])
        state_given_past_probs *= likelihoods
        ''' Assign to Nearest '''
        assignment_indices = np.argmax(state_given_past_probs, axis=1)
        super_future_matrix = np.tile(self.state_mean_futures_array, [pasts.shape[0],1,1])        
        return np.expand_dims(super_future_matrix[np.arange(len(assignment_indices)), assignment_indices], axis=2)

    def predict_batch_avg(self, pasts, points_label=None, do_not_cache=False):
        ''' Uses weighted average of states '''
        state_given_past_probs = []
        for k, s in enumerate(self.states):
            if self.verbose: print "Evaluating state", k
            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
        ''' Weight by state likelihood '''
        likelihoods = np.array([s.state_likelihood() for s in self.states])
        state_given_past_probs *= likelihoods
        state_given_past_probs = np.nextafter(state_given_past_probs, 1.)
        ''' Normalize '''
        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)        
        ''' Return weighted average '''
        return np.dot(state_given_past_probs, self.state_mean_futures_array)

    def predict_batch(self, pasts, points_label=None, do_not_cache=False):
        ''' Batch prediction of pasts '''
        return self.predict_batch_avg(pasts, points_label, do_not_cache)

    def compute_likelihoods(self, pasts, futures):
        state_given_past_probs = []
        future_given_state_probs = []
        for k, s in enumerate(self.states):
            if self.verbose: print "Evaluating state", k
            state_given_past_probs.append(s.batch_lc_likelihood_given_state(pasts))
            future_given_state_probs.append(s.batch_emission_likelihood_given_state(futures))
        state_given_past_probs = np.nextafter(state_given_past_probs, 1.).T
        future_given_state_probs = np.nextafter(future_given_state_probs, 1.).T
        ''' Weight by state likelihood '''
        likelihoods = np.array([s.state_likelihood() for s in self.states])
        state_given_past_probs *= likelihoods
        state_given_past_probs = np.nextafter(state_given_past_probs, 1.)
        ''' Normalize '''
        state_given_past_probs /= np.expand_dims(np.sum(state_given_past_probs, axis=1), axis=1)        
        ''' Return mixed likelihoods '''
        return np.nextafter(np.sum(np.multiply(state_given_past_probs, future_given_state_probs), axis=1), 1.)

    def total_prediction_MSE(self, pasts, true_futures):
        pasts = multi_flatten(pasts)
        predictions = self.predict_batch(pasts)
        truth = np.array(true_futures)
        return np.sum((predictions - truth)**2) / len(true_futures)

    def save_params(self, fold):
        pass

    def print_state_info(self):
        for s in self.states:
            print s.mean_future


def main():
    p1 = np.random.random((800000,4,3))
    p2 = np.random.random((200000,4,3)) + 8
    points = np.vstack((p1, p2))
    np.random.shuffle(points)
    ms = OHP(K_max=15, verbose=False)
    t1 = time()    
    ms.learn(points[:,:-1], points[:,-1])            
    print "Total Prediction MSE:", ms.total_prediction_MSE(points[:10,:-1], points[:10,-1])
    print "All done %0.2f minutes" % ((time() - t1) / 60)

if __name__ == "__main__":
    main()
