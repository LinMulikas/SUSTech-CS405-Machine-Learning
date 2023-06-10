import numpy as np


class LDA:
    #n_components: Number of components (<= min(n_classes - 1, n_features)) for dimensionality reduction.
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X:np.ndarray, y:np.ndarray):
        if self.n_components is None or self.n_components > X.shape[1]:
            n_components = X.shape[1]
        else:
            n_components = self.n_components
       
        n = np.shape(X)[1]
        labels = np.unique(y)
        Cs = []
        
        for lb in labels:
            Cs.append(np.argwhere(y == lb).reshape(-1).tolist()) 
            
        cls_indices = np.array(Cs)
        
        # k 类
        k = len(Cs)
        u_ks = []
        for i in range(k):
            u_ks.append(np.mean(X[cls_indices[i]], axis=0))
        # Within class scatter matrix
        # Complete code for calculating S_W
        ########### Write Your Code Here ###########
        S_W = np.zeros((n, n)) 
        for i in range(len(Cs)):
            S_k = np.zeros((n, n))
            for xi in X[cls_indices[i]]:
                vec: np.array = xi - u_ks[i]
                if(vec.ndim == 1):
                    vec = np.reshape(vec, (vec.shape[0], 1))
                
                S_k += vec @ vec.T
                                            
            S_W += S_k
        # Between class scatter matrix
        # Complete code for calculating S_B
        ########### Write Your Code Here ###########
        u = np.mean(u_ks, axis=0)
        S_B = np.zeros((n, n))
        for i in range(len(Cs)):
            vec = u_ks[i] - u
            if(vec.ndim == 1):
                vec = np.expand_dims(vec, axis=0)
            S_B += len(cls_indices[i])*(vec.T*vec)

      
        # Determine SW^-1 * SB by calculating inverse of SW
        ########### Write Your Code Here ###########
        A = np.linalg.inv(S_W) @ S_B 
        
        
        # Get eigenvalues and eigenvectors of SW^-1 * SB
        ########### Write Your Code Here ###########
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Sort the eigenvalues and corresponding eigenvectors from largest
        # to smallest eigenvalue and select the first n_components
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx][:n_components]
        eigenvectors = eigenvectors[:, idx][:, :n_components]
  
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
  
    def fit_transform(self, X):
        ########### Write Your Code Here ###########
        
        return None  
  
    def transform(self, X):
        ########### Write Your Code Here ###########
        return X@self.eigenvectors  

            
            