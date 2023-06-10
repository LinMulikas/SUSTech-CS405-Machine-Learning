from math import sqrt

import numpy as np

n, m, d = input().split(" ")
N = int(n)
M = int(m)
D = int(d)

X_train = np.zeros((N, D), dtype=float)
y_train = np.zeros((N,), dtype=int)
X_val = np.zeros((M, D), dtype=float)
y_val = np.zeros((M,), dtype=int)

for i in range(N):
    strs = input().split(" ")
    for d in range(D):
        X_train[i, d] = float(strs[d])
        
    y_train[i] = int(strs[D])
    

for i in range(M):
    strs = input().split(" ")
    for d in range(D):
        X_val[i, d] = float(strs[d])
        
    y_val[i] = int(strs[D])
    

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
            
            
    return np.array(lst)


def Majority(neighbors):
    """
    Performs majority voting and returns the predicted value for the test sample.
    
    Since we're performing binary classification the possible values are [0,1].
    
    Args:
        neighbors (list): K-nearest neighbors' labels
        
    Returns:
        predicted_value (int): predicted label for the given sample
    """
    
    # Performs majority voting
    # Complete this function
    res = np.argmax(np.bincount(neighbors))
    
    return res


def KNN(K, metric_type, X_train, y_train, X_val):
    """
    Returns the predicted values for the entire validation or test set.
    
    Please DO NOT use Scikit's KNN model when implementing this function. 

    Args:
        K (int): K-value
        metric_type (str): metric type
        X_train ((n,p) np.ndarray): Training data with n samples and p features
        y_train : Training labels
        X_val ((n, p) np.ndarray): Validation or test data
        
    Returns:
        predicted_values (list): output for every entry in validation/test dataset 
    """
    
    # Complete this function
    # Loop through the val_data or the test_data (as required)
    # and compute the output for every entry in that dataset  
    # You will also call the function "Majority" here
    
    # K = min(K, X_train.shape[0])
    K_nearest = computeDistancesNeighbors(K, metric_type, X_train, y_train, X_val)

    predictions = np.zeros((X_val.shape[0],), dtype='int64')

    for i in range(X_val.shape[0]):
        predictions[i] = Majority(K_nearest[i])
        

    return predictions


def evaluation(y_true, y_pred):
        """
        Computes the accuracy of the given datapoints.
        
        Args:
            predicted_values ((n,) np.ndarray): Predicted values for n samples
            actual_values ((n,) np.ndarray): Actual values for n samples
        
        Returns:
            accuracy (float): accuracy
        """
        cnt = 0
        n = y_true.shape[0]
        for i in range(n):
            if(y_true[i] == y_pred[i]):
                cnt += 1
                
        return cnt/n
    
    
res_mat = np.zeros((5, 3))

K = range(1, 6)
Norms = ["L1", "L2", "L-inf"]

max_val = 0
for k in K:
    for n in range(3):
        res_mat[k - 1, n] = evaluation(y_val, KNN(k, Norms[n], X_train, y_train, X_val))
        if(res_mat[k - 1, n] > max_val):
            max_val = res_mat[k - 1, n]
         
         
indices = []
for i in range(res_mat.shape[0]):
    for j in range(res_mat.shape[1]):
        if(res_mat[i][j] == max_val):
            indices.append((i, j))

         
for tup_index in indices:
    print(K[tup_index[0]],end = "")
    print(" ", end="")
    print(Norms[tup_index[1]])