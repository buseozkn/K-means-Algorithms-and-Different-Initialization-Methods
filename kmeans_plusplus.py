#K- means++
import numpy as np
import random
import matplotlib.pyplot as plt

def kmeans_pp(X,K):
    m,n= X.shape
    idx=np.random.randint(0,m)
    centroids=np.empty((K,n))
    first_centroid=X[idx]
    centroids[0,:]=first_centroid
    
    elements=range(0,m)
    probability_matrix=np.empty(m)
    
    k=1
    while k<K:
        differences= X[:, np.newaxis, :]-centroids
        distances_to_centroids = np.linalg.norm(differences, axis=2)
        distances_to_nearest_centroids=np.min(distances_to_centroids, axis=1)
    
        probability_matrix=((distances_to_nearest_centroids)**2)/np.sum((distances_to_nearest_centroids)**2)
        
        weights = list(probability_matrix)
        selected_centroid = random.choices(elements, weights=weights, k=1)
        centroids[k,:]=X[selected_centroid,:]
        k+=1
    
    return centroids
        
        

"""X=np.random.rand(200,2)
centroids=kmeans_pp(X,5)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=100, label='Centroids', marker='x')
plt.title('Data Points and Initialized Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()"""