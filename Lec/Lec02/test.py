from cmath import sqrt

import numpy as np


def distanceFunc(metric_type, vec1, vec2):
    """
    Computes the distance between two d-dimension vectors. 
    
    Please DO NOT use Numpy's norm function when implementing this function. 
    
    Args:
        metric_type (str): Metric: L1, L2, or L-inf
        vec1 ((d,) np.ndarray): d-dim vector
        vec2 ((d,)) np.ndarray): d-dim vector
    
    Returns:
        distance (float): distance between the two vectors
    """

    diff = vec1 - vec2
    if metric_type == "L1":
        summ = 0
        for i in range(diff.shape[0]):
            summ += abs(diff[i])
        
        distance = summ

    if metric_type == "L2":
        summ = 0
        n = diff.shape[0]
        for i in range(n):
            summ += diff[i]**2
            
        distance = sqrt(summ/n).real
        
    if metric_type == "L-inf":
        for i in range(diff.shape[0]):
            diff[i] = abs(diff[i])
            
        distance = max(diff)
        
    return distance

def computeDistancesNeighbors(K, metric_type, X_train, y_train, sample):
    """
    Compute the distances between every datapoint in the train_data and the 
    given sample. Then, find the k-nearest neighbors.
    
    Return a numpy array of the label of the k-nearest neighbors.
    
    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        sample ((p,) np.ndarray): Single sample whose distance is to computed with every entry in the dataset
        
    Returns:
        neighbors (list): K-nearest neighbors' labels
    """

    # You will also call the function "distanceFunc" here
    # Complete this function
    (m, n) = X_train.shape
    p = sample.shape[0]
    
    dist = np.zeros((p, m))
    
    for i in range(0, p):
        for j in range(0, m):
            dist[i][j] = distanceFunc(metric_type, sample[i], X_train[j])
            
        
    indices = np.argsort(dist, kind='stable')
    lst = []
    
    for i in range(p):
        lst.append(y_train[indices[i, :K]])
            
            
    return lst


K = 1
metric_type = "L2"
X_train = np.array([[1, 2, 3], 
                    [1, 2, 4],
                    [1, 2, 6]])
y_train = np.array([0, 1, 1])

sample = np.array([[1, 2, 5]])
computeDistancesNeighbors(K, metric_type, X_train, y_train, sample)