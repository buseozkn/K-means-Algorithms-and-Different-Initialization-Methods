#K-medians
import sys
sys.path.append("C:/Users/Lenovo/Desktop/K-MEANS")
import matplotlib.pyplot as plt
import kaufman as kaufman
import maximin as maximin
import kmeans_plusplus as kmeans_pp
import numpy as np

def Kmedians(X,K):
    X=np.array((X))
    m,n=X.shape
    centroids=kmeans_pp.kmeans_pp(X,K)
    previous_centroids=np.empty((K,n))
    
    while not np.array_equal(centroids, previous_centroids):
        previous_centroids=np.copy(centroids)
        
        #X[:, np.newaxis, :] has shape (m, 1, n)
        #differences has shape (m, K, n)
        differences= X[:, np.newaxis, :]-centroids
        
        #The result will be of shape (m, K)
        distances_to_centroids = np.linalg.norm(differences, axis=2)
        centroid_assignments=np.argmin(distances_to_centroids, axis=1)

        centroids_dict={}
        for i in range(K):
            centroids_dict[i]=[]
        
        for datapoint in range(m):
            centroids_dict[centroid_assignments[datapoint]].append(X[datapoint])
        
        for centroid in range(len(centroids_dict)):
            centroid_array=np.array(centroids_dict[centroid])
            centroids[centroid]=np.median(centroid_array)
            
    ssd=0
    for data in range(m):
        ssd+=np.linalg.norm(X[data]-centroids[centroid_assignments[data]])** 2
        
    
    return centroids,centroid_assignments, ssd

np.random.seed(100)
X=np.random.rand(200,2)

ssd_matrix=np.empty((7))
for K in range(3,10):
    _,_,ssd=Kmedians(X,K)
    ssd_matrix[K-3]=ssd
    
 
plt.title("SSD vs Number of Clusters")    
plt.plot(range(3,10),ssd_matrix)
plt.xlabel('Number of Clusters')
plt.ylabel('SSD')
plt.grid(True)
plt.show()


K=4
centroids, assignments,_=Kmedians(X,K)

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Extend this list for more than 7 clusters

for k in range(K):
    cluster_points = X[assignments == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors[k])
    plt.scatter(centroids[k, 0], centroids[k, 1], s=200, color=colors[k], edgecolors='k', marker='*')

plt.title('K-medians Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
