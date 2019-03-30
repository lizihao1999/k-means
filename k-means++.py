# complete k-means++
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
#load data
data = np.loadtxt('Downloads/mnist_small.txt')
#normalize the data 
data=np.divide(data,16)
#distance function
def dist(a, b, ax=1):
    return np.linalg.norm(b - a, axis=ax)
#20 iterations
for k in range(20):
    #initialize the centroids 
    center=np.zeros((10,64))
    temp=np.zeros(1797)
    index=np.random.choice(1797)
    center[0]=data[index]
    for i in range(1,10):
        for j in range(1797):
            temp[j]=(np.amin(dist(data[j],center[0:i])))**2
        temp=np.divide(temp,np.sum(temp))   
        index=np.random.choice(1797,p=temp)
        center[i]=data[index]

    center_old = np.zeros(center.shape)

    clusters = np.zeros(data.shape[0])

    error = dist(center, center_old, None)

    sum=[]

    while error!=0:
        s=0
        #assign each datapoint to cluster
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

plt.savefig('Downloads/kmeans++ distortion')
plt.show()

#center=center*16
#plt.gray()
#for i in range(10):
#    plt.imshow(center[i].reshape(8,8))
#    plt.savefig('Downloads/centroid_++(%d).png' % i)
#    plt.show()
    
