#complete k-means, 20 iterations, with centoirds and distortions plot

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
#load data
data = np.loadtxt('Downloads/mnist_small.txt')

#divided all values by 16 to normalize
data=np.divide(data,16)

#distance function
def dist(a, b, ax=1):
    return np.linalg.norm(b - a, axis=ax)

#20 iterations
for k in range(20):
    #random centroids
    center=np.random.rand(10,64)
    #an arrray to record the old centroids
    center_old = np.zeros(center.shape)
    #store the clusters indexes  
    clusters = np.zeros(data.shape[0])
    #determine whether centroids change
    error = dist(center, center_old, None)
    
    sum=[]

    while error!=0:
        s=0
        #assign each datapoints to ideal clusters
        for i in range (1797):
            distances = dist(data[i], center)
            cluster = np.argmin(distances)
            s=s+distances[cluster]
            clusters[i] = cluster
        sum.append(s)
        center_old = deepcopy(center)
        #re-compute the centroids
        for i in range(10):
            points = [data[j] for j in range (1797) if clusters[j] == i]
            if len(points)!=0:
                center[i] = np.mean(points, axis=0)
        error=dist(center, center_old, None)

    x=range(len(sum))
    plt.plot(x,sum)

plt.savefig('Downloads/kmeans distortion')
plt.show()
    
#center=center*16
#plt.gray()
#for i in range(10):
#    plt.imshow(center[i].reshape(8,8))
#    plt.savefig('Downloads/centroid(%d).png' % i)
#    plt.show()
