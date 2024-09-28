import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform, cdist
import math

def local_density(X):
    
    pairwise_distances = pdist(X, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    mst = minimum_spanning_tree(distance_matrix)
    mst_dense = mst.toarray()
    weights_mst = mst_dense[mst_dense != 0].flatten()

    Q1 = np.percentile(weights_mst, 25)
    Q3 = np.percentile(weights_mst, 75)
    IQR = Q3 - Q1
    epsilon=3* IQR + Q3

    pairwise_distances = pdist(X, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, np.inf)
    filter=distance_matrix <= epsilon
    neighbor_matrix=-(filter*distance_matrix)/epsilon

    small_value = math.exp(-100)
    neighbor_matrix[neighbor_matrix== 0] = small_value

    local_d=np.sum(neighbor_matrix, axis=1)
    return local_d

def dk_means(X,K):

    local_d=local_density(X)
    first_centroid_index=np.argmax(local_d)
    first_centroid=X[first_centroid_index]
    centroids=first_centroid
    X= np.delete(X, first_centroid_index, axis=0)
    k=2

    while k<=K:
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)

        distances = cdist(X, centroids, metric='euclidean')
        nearest_centroids = np.min(distances, axis=1)
        
        local_d=local_density(X)
        prospectiveness=local_d*nearest_centroids
        centroid_index=np.argmax(prospectiveness)
        centroid = X[centroid_index]

        X= np.delete(X, centroid_index, axis=0)
        centroids=np.vstack((centroids, centroid))
        k+=1
        return centroids

