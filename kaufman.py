#Kaufmann
import numpy as np
import matplotlib.pyplot as plt

def kaufman(X,K):
    #X: np array (m, n) m: number of data points, n: number of features
    m,n=X.shape
    globalcentroid=np.mean(X, axis=0)
    global_centroid_matrix=np.tile(globalcentroid, (m,1))
    distances_to_global=np.sum((X-global_centroid_matrix)**2,axis=1)**0.5 #(m, 1) array
    thefirst_centroid_index=np.argmin(distances_to_global)
    thefirst_centroid=X[thefirst_centroid_index,:]
    centroids=np.empty((K,n))
    centroids[0,:]=thefirst_centroid
    A=X
    A=np.delete(A,thefirst_centroid_index,0)
    Cii_matrix=np.empty((A.shape[0],A.shape[0]))

    k=2
    while k <=K:
        for row in range(Cii_matrix.shape[0]):
            D=np.empty((K))
            for centroid in range(centroids.shape[0]):
                D[centroid]=np.sum((A[row,:]-centroids[centroid])**2)
            Dxi=np.min(D)
            for column in range(Cii_matrix.shape[1]):
                if row!=column:
                    distance= np.sum((A[row,:]-A[column,:])**2)
                    Cii_matrix[row,column]=Dxi-distance
                    if Cii_matrix[row,column]<0:
                        Cii_matrix[row,column]=0
        sumCii_vector=np.sum(Cii_matrix, axis=1)
        centroids[k-1,:]=A[np.argmax(sumCii_vector)]
        A=np.delete(A,np.argmax(sumCii_vector),0)
        Cii_matrix=np.delete(Cii_matrix,np.argmax(sumCii_vector),0)
        Cii_matrix=np.delete(Cii_matrix,np.argmax(sumCii_vector),1) 
        k+=1                                          
    
    return centroids


    
"""X=np.random.rand(10,2)
centroids=kaufman(X,5)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data Points')
plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=100, label='Centroids', marker='x')
plt.title('Data Points and Initialized Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()"""