import random
import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist


def LOF(X,mp,m):
    pairwise_distances = pdist(X, metric='euclidean')
    distance_matrix = squareform(pairwise_distances)
    np.fill_diagonal(distance_matrix, np.inf)
    smallest_mp_per_row = np.argsort(distance_matrix, axis=1)[:, :mp]
    distances_mp=np.zeros(m)
    for k in range(m):
        indices_row=smallest_mp_per_row[k]
        distance=np.sum(np.sqrt(np.sum((X[k]-X[indices_row])**2, axis=1)))
        distances_mp[k]=distance
    density=mp/distances_mp

    densities=np.zeros(m)
    for k in range(m):
        indices_row=smallest_mp_per_row[k]
        densities[k]=np.sum(density[indices_row])

    ard=density/(densities/mp)
    LOF=1/ard
    return LOF

def ROBIN(X,K,m,mp):
    ref_point=X[random.randint(0,m)]
    distance_matrix=np.sqrt(np.sum((ref_point-X)**2, axis=1))
    sorted_indices = np.argsort(distance_matrix)[::-1]
    epsilon=0.05
    LOF_=LOF(X,mp,m)
    Sorted_LOF=LOF_[sorted_indices]
    filtered_LOF1=Sorted_LOF <= 1 + epsilon 
    filtered_LOF2=Sorted_LOF >= 1 - epsilon
    filtered_LOF=filtered_LOF1* filtered_LOF2
    first_centroid = np.where(filtered_LOF == True)[0][0]
    k=1
    centroids=X[first_centroid]
    X= np.delete(X, first_centroid, axis=0)
    while k<=K:
        if centroids.ndim == 1:
            centroids = centroids.reshape(1, -1)
        distances = cdist(X, centroids, metric='euclidean')
        nearest_centroids = np.min(distances, axis=1)
        sorted_indices2 = np.argsort(nearest_centroids)[::-1]
        Sorted_LOF=LOF_[sorted_indices2]
        filtered_LOF1=Sorted_LOF <= 1 + epsilon 
        filtered_LOF2=Sorted_LOF >= 1 - epsilon
        filtered_LOF=filtered_LOF1* filtered_LOF2
        centroid = np.where(filtered_LOF == True)[0][0]
        X= np.delete(X, centroid, axis=0)
        centroids=np.vstack((centroids, X[centroid]))
        k+=1
        return centroids


