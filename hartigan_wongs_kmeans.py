#Hartigan Wong' s K-means
import sys
sys.path.append("C:/Users/Lenovo/Desktop/K-MEANS")
import matplotlib.pyplot as plt
import kaufman as kaufman
import maximin as maximin
import kmeans_plusplus as kmeans_pp
import numpy as np

def hartigan_wongs(X,K):
    X=np.array((X))
    m,n=X.shape
    
    #cluster assignment
    initial_centroids=kmeans_pp.kmeans_pp(X,K)
    differences= X[:, np.newaxis, :]-initial_centroids
    distances_to_centroids = np.linalg.norm(differences, axis=2)
    centroid_assignments=np.argmin(distances_to_centroids, axis=1)
    
    #centroid calculation
    numbers_in_clusters=np.zeros((K,1))
    for row in range(m):
        numbers_in_clusters[centroid_assignments[row],:]+=1
    
    centroids=np.copy(initial_centroids)
    j=1
    while j==1:
        
        s=1
        for datapoint in range(m):

            
            which_centroid=centroid_assignments[datapoint]
            temporary_centroids=np.copy(centroids)
            
            #centroid calculation excluding centroid_assignments[datapoint]
            if numbers_in_clusters[which_centroid]-1>=1:
                temporary_centroids[which_centroid,:]=(centroids[which_centroid,:]*numbers_in_clusters[which_centroid]-X[datapoint,:])/(numbers_in_clusters[which_centroid]-1)
            
            #cluster assignment
            temporary_differences= X[datapoint]-temporary_centroids
            temporary_distances_to_centroids = np.linalg.norm(temporary_differences, axis=1)
            new_centroid=np.argmin(temporary_distances_to_centroids)
            
            #centroid calculation
            if new_centroid!=which_centroid:
                centroid_assignments[datapoint]=new_centroid
                numbers_in_clusters[which_centroid]-=1
                numbers_in_clusters[new_centroid]+=1

                if numbers_in_clusters[which_centroid]>0:
                   centroids[which_centroid]= (centroids[which_centroid]* (numbers_in_clusters[which_centroid] + 1) - X[datapoint,:]) / numbers_in_clusters[which_centroid] 
            
                
                centroids[new_centroid]= (centroids[new_centroid]* (numbers_in_clusters[new_centroid] - 1) + X[datapoint,:]) / numbers_in_clusters[new_centroid]
                centroid_assignments[datapoint]= new_centroid
            
                s=0
        if s==1:
            j=0

            
    ssd=0
    for data in range(m):
        ssd+=np.linalg.norm(X[data]-centroids[centroid_assignments[data]])** 2
        
    return centroids, centroid_assignments, ssd

np.random.seed(100)
X=np.random.rand(200,2)

ssd_matrix=np.empty((7))
for K in range(3,10):
    _,_,ssd=hartigan_wongs(X,K)
    ssd_matrix[K-3]=ssd
    
 
plt.title("SSD vs Number of Clusters")    
plt.plot(range(3,10),ssd_matrix)
plt.xlabel('Number of Clusters')
plt.ylabel('SSD')
plt.grid(True)
plt.show()


K=4
centroids, assignments,_=hartigan_wongs(X,K)

plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']  # Extend this list for more than 7 clusters

for k in range(K):
    cluster_points = X[assignments == k]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=30, color=colors[k])
    plt.scatter(centroids[k, 0], centroids[k, 1], s=200, color=colors[k], edgecolors='k', marker='*')

plt.title('Hartigan Wong\'s K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()