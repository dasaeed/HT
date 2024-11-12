import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, loggamma

ETA = 0.5 # Hyperparameter for V-Dirichlet topics prior
ALPHA = 0.5 # Hyperparameter for K-Dirichlet topics proportion prior

def sim_LDA(K, V, N, M):
    """
    Generate a collection of documents based on LDA model.

    K: Number of topics
    V: Number of terms in vocabulary
    N: Number of documents
    M: Number of words in each document
    """

    # Draw topic distribution for each k = 1,...,K by V-Dirichlet(ETA, K)
    BETA = np.random.dirichlet(ETA * np.ones(V), K)

    X = [] # Initialize empty list for complete collection of documents
    for i in range(N): # Iterate over number of documents i = 1,...,N
        # Draw topic proportions by K-Dirichlet(ALPHA)
        THETA = np.random.dirichlet(ALPHA * np.ones(K), N)

        x = []
        for _ in range(M): # Iterate over each word (word count is constant across documents)
            z_ij = np.random.choice(K, p=THETA[i]) # Draw topic assignment by Multinom(THETA_d)
            x_ij = np.random.choice(V, p=BETA[z_ij]) # Draw observed word by Multinom(BETA_(z_ij))
            x.append(x_ij)
        X.append(x)

    X = np.asarray(X) # Convert collection of documents to numpy array
    return X

def init_variation_params(X):
    """
    Initialize variational parameters for LDA model.

    X: Collection of documents
    K: Number of topics
    V: Number of terms in vocabulary
    """

    N, M = X.shape # Get the number of documents and the number of words per document

    # Random initialization for variational topics LAMBDA (uniform prior)
    LAMBDA = np.random.uniform(low=0.01, high=1.00, size=(K, V))

    # Initialize variational topic proportions to 1 (maybe random initialization?)
    GAMMA = np.ones(shape=(N, K))

    # Initialize variational topic assignments to 1/K
    PHI = np.ones(shape=(N, M, K)) * 1/K

    return LAMBDA, GAMMA, PHI

def compute_ELBO(LAMBDA, GAMMA, PHI, X):
    """
    Compute the ELBO for an LDA model given variational parameters (LAMBDA, GAMMA, PHI).
    The computation is split between expected log joint terms and entropy terms. 

    LAMBDA: Variational topic distribution for BETA
    GAMMA: Variational document topic proportion's distribution for THETA
    PHI: Variational categorical topic assignments
    X: LDA-simulated collection of documents
    """

    N, M = X.shape # Get number of documents (N) and number of words per document (M)
    ELBO = 0 # Initialize ELBO

    # EXPECTED LOG JOINT TERMS

    # Term is \sum_{i=1}^K E_q[log p(BETA_k; ETA)]
    E_log_p_BETA = 0
    for k in range(K):
        E_log_p_BETA += (ETA - 1) * np.sum(digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k])))
    ELBO += E_log_p_BETA

    # Term is \sum_{i=1}^N E_q[log p(THETA_i; ALPHA)]
    E_log_p_THETA = 0
    for i in range(N):
        E_log_p_THETA += (ALPHA - 1) * np.sum(digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i])))
    ELBO += E_log_p_THETA

    # Term is \sum_{i=1}^N \sum_{j=1}^M (E_q[log p(z_ij | THETA_i)] + E_q[log p(x_ij | BETA, z_ij)])
    E_q_log_p_z_x = 0
    for i in range(N):
        for j in range(M):
            # For E_q[log BETA_{k, x_ij}] term, x_{ij} is some assignment from vocabulary v=1,...,V
            x_ij = X[i,j]
            E_q_log_p_z_x += np.sum(PHI[i,j] * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i])))) \
                + np.sum(PHI[i,j] * (digamma(LAMBDA[:, x_ij])) - digamma(np.sum(LAMBDA[:, x_ij], axis=0)))
    ELBO = E_q_log_p_z_x

    # ENTROPY TERMS

    # Term is -\sum_{k=1}^K E_q[log q(BETA_k; LAMBDA_k)]
    E_log_q_BETA = 0
    for k in range(K):
        E_log_q_BETA += -loggamma(np.sum(LAMBDA[k])) + np.sum(loggamma(LAMBDA[k])) \
            - np.sum((LAMBDA[k] - 1) * (digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k]))))
    ELBO += E_log_q_BETA

    # Term is -\sum_{i=1}^N E_q[log q(THETA_i; GAMMA_i)]
    E_log_q_THETA = 0
    for i in range(N):
        E_log_q_THETA += -loggamma(np.sum(GAMMA[i])) + np.sum(loggamma(GAMMA[i])) \
            - np.sum((GAMMA[i] - 1) * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))
    ELBO += E_log_q_THETA

    # Term is -\sum_{i=1}^n \sum_{j=1}^M E_q[log q(z_ij; PHI_ij)]
    E_q_log_z = 0
    for i in range(N):
        for j in range(M):
            E_q_log_z += -np.sum(PHI[i,j] * np.log(PHI[i,j]))
    ELBO += E_q_log_z

    return ELBO


def log_sum_exp(vec):
    """"
    Log-sum-exponential trick. Used to normalize the categorical 
    distribution of PHI[i, j] in CAVI updates.
    """

    alpha = np.max(vec, axis=0)
    log_sum_exp = np.log(np.sum(np.exp(vec - alpha))) + alpha

    return log_sum_exp

########## EXAMPLE ##########

K = 10 # Number of topics
V = 50 # Number of terms in the vocabulary
N = 20 # Number of documents
M = 30 # Number of words per document
ETA = 0.5 # Hyperparameter for V-Dirichlet topics prior
ALPHA = 0.5 # Hyperparameter for K-Dirichlet topics proportion prior
X = sim_LDA(K, V, N, M) # Simulate LDA-collection of documents

start = time.time()
LAMBDA, GAMMA, PHI = init_variation_params(X)
curr_ELBO = 100 # Initialize current ELBO value (initially arbitrarily greater than prev_ELBO)
prev_ELBO = 0 # Initialize previous ELBO value (initially arbitrarily smaller than curr_ELBO)
ELBOs = [] # Store ELBO values for plotting
tol = 10e-10 # Tolerance for convergence of ELBO

# Continue to update variational parameters until ELBO has converged with respect to tolerance (tol)
while np.abs(curr_ELBO - prev_ELBO) > tol:
    # Update variational topic assignment PHI[i, j]
    for i in range(N):
        for j in range(M):
            # Again, x_ij is just an assignment of the vocabulary v=1,...,V so can treat it as such
            x_ij = X[i, j]

            # Calculate the expression inside the exponential for which PHI[i, j] is proportional to
            exp_propto = digamma(LAMBDA[:, x_ij]) - digamma(np.sum(LAMBDA[:, x_ij])) \
                + digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))
            
            # Use log-sum-exp trick to normalize PHI[i, j] over k
            PHI[i, j] = np.exp(exp_propto - log_sum_exp(exp_propto))

        # Update variational topic proportions GAMMA[i]
        for k in range(K):
            # Update is ALPHA + sum_{j=1}^M PHI_ijk
            GAMMA[i, k] = ALPHA + np.sum(PHI[i][:, k])

    # Update variational topic LAMBDA[k]
    for k in range(K):
        for v in range(V):
            # Update is ETA + \sum_{i=1}^N \sum_{j=1}^M 1{x_ij = v} PHI_ijk
            LAMBDA[k, v] = ETA + np.sum([[float(X[i,j] == v) * PHI[i, j][k] for i in range(N)] for j in range(M)])
    
    prev_ELBO = curr_ELBO # Set the previous ELBO to the current value of previous iteration
    curr_ELBO = compute_ELBO(LAMBDA, GAMMA, PHI, X) # Compute ELBO for updated parameters; set current ELBO to new ELBO
    ELBOs.append(curr_ELBO) # Store computed ELBO values

total_time = time.time() - start

fig, axes = plt.subplots()
axes.set_xlabel("Seconds")
axes.set_ylabel("ELBO")
axes.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

axes.plot(np.linspace(0, total_time, len(ELBOs)), ELBOs)