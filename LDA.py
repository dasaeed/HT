from tqdm import tqdm 
import copy
import numpy as np
from scipy.special import digamma, loggamma

def sim_LDA(K, V, N, M, ETA, ALPHA):
    """
    Generate a collection of documents based on LDA model.

    K: Number of topics
    V: Number of terms in vocabulary
    N: Number of documents
    M: Number of words in each document
    ETA: Hyperparameter for V-Dirichlet topics prior
    ALPHA: Hyperparameter for K-Dirichlet topics proportion prior
    """

    # Draw topic distribution for each k = 1,...,K by V-Dirichlet(ETA, K)
    BETA = np.random.dirichlet(ETA * np.ones(V), K)

    X = [] # Initialize empty list for complete collection of documents
    for i in range(N): # Iterate over number of documents i = 1,...,N
        # Draw topic proportions by K-Dirichlet(ALPHA)
        THETA = np.random.dirichlet(ALPHA * np.ones(K), N)

        x = []
        for _ in range(M): # Iterate over each word
            z_ij = np.random.choice(K, p=THETA[i]) # Draw topic assignment by Multinom(THETA_d)
            x_ij = np.random.choice(V, p=BETA[z_ij]) # Draw observed word by Multinom(BETA_(z_ij))
            x.append(x_ij)
        X.append(x)

    X = np.asarray(X) # Convert collection of documents to numpy array
    return X