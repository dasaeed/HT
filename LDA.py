from tqdm import tqdm 
import copy
import numpy as np
from scipy.special import digamma, loggamma

def sim_LDA(K, V, N, M):
    # For each topic in k = 1,...,K, draw topic distributions
    BETA = np.random.dirichlet(ETA * np.ones(V), K)

    X = [] # Initialize empty list for complete collection of documents
    for i in range(N): # Iterate over number of documents i = 1,...,N
        THETA = np.random.dirichlet(ALPHA * np.ones(K), N) # Draw topic proportions
        x = []
        for _ in range(M): # Iterate over each word
            z_ij = np.random.choice(K, p=THETA[i]) # Draw topic assignment
            x_ij = np.random.choice(V, p=BETA[z_ij]) # Draw observed word
            x.append(x_ij)
        X.append(x)

    X = np.asarray(X) # Convert collection of documents to numpy array
    return X