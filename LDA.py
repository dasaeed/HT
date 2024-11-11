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

def init_variation_params(X, K, V):
    """
    Initialize variational parameters for LDA model.

    X: Collection of documents
    K: Number of topics
    V: Number of terms in vocabulary
    """

    N, M = X.shape # Get the number of documents and the number of words per document

    # Random initialization for variational topics LAMBDA
    LAMBDA = np.random.uniform(low=0.01, high=1.00, size=(K, V))

    # Initialize variational topic proportions to 1
    GAMMA = np.ones(shape=(N, K))

    # Initialize variational topic assignments to 1/K
    PHI = np.ones(shape=(N, M, K)) * 1/K

    return LAMBDA, GAMMA, PHI