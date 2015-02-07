from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN 
from sklearn.preprocessing import StandardScaler

def merge_trees(tree, old_val, new_val):
    for node, cluster in tree.iteritems():
        if cluster == old_val:
            tree[node] = new_val            

def build_mst(adj_matrix):
    block_size = 1000
    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    subtrees = {}
    tree = []
    edges = []
    edges.extend([(adj_matrix[i][j],i,j) for i in range(rows) for j in range(cols) if i != j])
    edges.sort()
    count = 0
    for w,i,j in edges:
        if not i in subtrees:
            subtrees[i] = -1
        if not j in subtrees:
            subtrees[j] = -2 
        if subtrees[i] != subtrees[j]:
            tree.append((i,j))
            if subtrees[i] == -1:
                subtrees[i] = count
                count += 1 
            merge_trees(subtrees, subtrees[j], subtrees[i])
    return tree

def cluster_seeds_dbscan(seeds, index_array, cluster_assignments, epsilon):
    ratio = 0.0
    while ratio < .9:
        db = DBSCAN(eps=epsilon, min_samples=3)
        res = np.array(db.fit_predict(seeds), dtype=int)
        cluster_assignments[index_array[:seeds.shape[0]]] = res
        ratio = np.sum(np.greater(cluster_assignments[index_array[:seeds.shape[0]]], -1)) / len(cluster_assignments[index_array[:seeds.shape[0]]])
        epsilon *= 1.25
        print "cluster coverage: %0.2f" % ratio, "(%0.3f)" % epsilon
    return cluster_assignments + (np.min(cluster_assignments) == -1) * 1

def cluster_seeds_gt(seeds, index_array, cluster_assignments):
    adj_matrix = cdist(seeds, seeds, 'euclidean')
    tree = build_mst(adj_matrix)
    edge_matrix = np.zeros((seeds.shape[0], seeds.shape[0]), dtype=int)
    for i, j in tree:
        edge_matrix[i, j] = 1
        edge_matrix[j, i] = 1
    ''' Remove locally abnormal edges '''        
    for row in range(edge_matrix.shape[0]):
        local_edge_lengths = adj_matrix[row, np.greater(edge_matrix[row,:], 0)].ravel()
        edge_indices = np.nonzero(edge_matrix[row,:])[0]
        if len(local_edge_lengths) > 2:
            for edge_counter in range(local_edge_lengths.shape[0]):
                masked_edges = np.ma.array(local_edge_lengths, mask=False)
                masked_edges.mask[edge_counter] = True                
                mean = np.mean(masked_edges)
                std = np.std(masked_edges)
                if np.abs(local_edge_lengths[edge_counter] - mean) > std * 3:
                    edge_matrix[row, edge_indices[edge_counter]] = 0
    ''' Form clusters by connected components '''
    cluster_map = {}
    for i in range(edge_matrix.shape[0]):
        current = index_array[i]
        connected_nodes = index_array[np.nonzero(edge_matrix[i])[0]]
        node_set = set(connected_nodes).union([current])
        for node in list(node_set):
            if node in cluster_map:
                node_set.update(cluster_map[node])
        for node in list(node_set):       
            cluster_map[node] = node_set    
    count = 0
    for i in range(edge_matrix.shape[0]):
        current = index_array[i]
        if cluster_assignments[current] == -1:
            cluster_assignments[current] = count
            count += 1
        cluster_assignments[list(cluster_map[current])] = cluster_assignments[current]
    return cluster_assignments

def cluster(instances, seed_method='dbscan', assign_to_nearest_center=True, cluster_eps=0.25):    
    BLOCK_SIZE = 1000
    N = instances.shape[0]
    MAX_SEEDS = 10000
    instances = instances.reshape((N, -1))
    cluster_assignments = np.repeat(-1, N)
    cluster_centers = {}
    index_array = np.arange(N)
    np.random.shuffle(index_array)
    seeds = instances[index_array[:MAX_SEEDS]]
    if seed_method == 'gt' and MAX_SEEDS <= 1000:
        cluster_assignments = cluster_seeds_gt(seeds, index_array, cluster_assignments)
    else:
        cluster_assignments = cluster_seeds_dbscan(seeds, index_array, cluster_assignments, cluster_eps)
    ''' Calculate cluster centers '''
    for c in set(cluster_assignments).difference([-1]):
        cluster_centers[c] = np.mean(instances[np.equal(cluster_assignments, c)], axis=0)
    ''' Add all other non-seed instances to clusters '''
    non_seeds = instances[index_array[MAX_SEEDS:]]
    if assign_to_nearest_center:
        ''' Assign remaining points to cluster with nearest center - FAST'''
        cluster_keys = np.array(sorted(cluster_centers.keys()))
        centers_array = np.vstack([cluster_centers[c] for c in cluster_keys])
        reference_pts = centers_array
    else:
        ''' Assign remaining points to clusters with nearest point - SLOW '''
        reference_pts = seeds
    start, end = 0, BLOCK_SIZE
    for block in range(non_seeds.shape[0] // BLOCK_SIZE):
        start = block * BLOCK_SIZE
        end = (block * BLOCK_SIZE) + BLOCK_SIZE
        non_seed_block = non_seeds[start : end]
        distance_array = cdist(non_seed_block, reference_pts, 'euclidean')
        start_offset = MAX_SEEDS + start
        end_offset = MAX_SEEDS + end
        if assign_to_nearest_center:
            cluster_assignments[index_array[start_offset:end_offset]] = cluster_keys[np.argmin(distance_array, axis=1)]
        else:
            cluster_assignments[index_array[start_offset:end_offset]] = cluster_assignments[np.argmin(distance_array, axis=1)]
    ''' Assign remaining clusters '''
    if MAX_SEEDS + end <= non_seeds.shape[0]:
        start_offset = end + MAX_SEEDS
        distance_array = cdist(non_seeds[start_offset:], reference_pts, 'euclidean')
        cluster_assignments[index_array[start_offset:]] = cluster_assignments[np.argmin(distance_array, axis=1)]
    return cluster_assignments
