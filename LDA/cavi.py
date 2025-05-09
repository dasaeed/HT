import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
from scipy.special import digamma, loggamma
import matplotlib.pyplot as plt
import time

# Computes the softmax for a given set of scores
# Scores will be log probability of each assignment
# Note: Uses log-sum-exp trick to avoid overflow/underflow
def compute_logsumexp(scores):
    # Take the max of the scores
    max_score = np.max(scores, axis=0)

    # Subtract scores by max and exponentiate
    exp_scores = np.exp(scores - max_score)

    # Compute denominator
    sum_exp = np.sum(exp_scores)

    # Compute log-sum-exp
    log_sum_exp = np.log(sum_exp) + max_score

    return log_sum_exp

# Initializes the variational parameters for CAVI
def init_var_param(train_articles, C):
    print('Initializing variational parameters...')

    # Number of articles, vocabulary size
    N, V = train_articles.shape

    # Topics (initializing LAMBDA for BETA)
    LAMBDA = np.random.uniform(low=0.01, high=1.0, size=(C,V))

    # Topic Proportions (initializing GAMMA for THETA)
    GAMMA = np.ones((N,C)) # Uniform prior

    # Topic Assignments (initializing PHI for Z)
    # Shape: (N,n_words,C) (Note: n_words is variable)
    PHI = []

    for article in train_articles:
        n_words = np.sum((article > 0).astype('int32'))
        article_phi = np.ones((n_words,C))
        article_phi = article_phi / C # Initialize to 1/C

        PHI.append(article_phi)

    return LAMBDA, GAMMA, PHI

# Compute ELBO
def compute_elbo(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C, ETA=0.01, ALPHA=0.1):
    elbo = 0

    # Number of articles, vocabulary size
    N, V = train_articles.shape

    # Add expected log joint
    ## First term: \sum_{k=1}^C E[log p(BETA_k)]
    E_log_p_beta = 0
    for k in range(C):
        E_log_p_beta += (ETA-1) * np.sum(digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k])))

    elbo += E_log_p_beta

    ## Second term: \sum_{i=1}^N E[log p(THETA_i)]
    E_log_p_theta = 0
    for i in range(N):
        E_log_p_theta += (ALPHA-1) * np.sum(digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i])))

    elbo += E_log_p_theta

    ## Third term:
    ## \sum_{i=1}^N \sum_{j=1}^M \sum_{k=1}^C
    ## (E[log p(Z_ij|THETA_i)] + E[log p(X_ij)|BETA,Z_ij)])
    E_log_p_xz = 0
    for i in range(N):
        article = train_articles[i]
        nonzero_idx = train_nonzero_idxs[i]

        corr_idx = 0

        for idx in nonzero_idx:
            ### E[log p(Z_ij|THETA_i)]
            E_log_p_xz += article[idx] * np.sum(PHI[i][corr_idx] * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))

            ### E[log p(X_ij|BETA,Z_ij)]
            E_log_p_xz += article[idx] * np.sum(PHI[i][corr_idx] * (digamma(LAMBDA[:,idx]) - digamma(np.sum(LAMBDA, axis=1))))

            corr_idx += 1

        # Check if number of updates match with number of words
        assert(corr_idx == len(nonzero_idx))

    elbo += E_log_p_xz

    # Add entropy
    ## Fourth term: -\sum_{k=1}^C E[log q(BETA_k)]
    E_log_q_beta = 0
    for k in range(C):
        E_log_q_beta += -loggamma(np.sum(LAMBDA[k])) + np.sum(loggamma(LAMBDA[k]))
        E_log_q_beta += -np.sum((LAMBDA[k]-1) * (digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k]))))

    elbo += E_log_q_beta

    ## Fifth term: -\sum_{i=1}^N E[log q(THETA_i)]
    E_log_q_theta = 0
    for i in range(N):
        E_log_q_theta += -loggamma(np.sum(GAMMA[i])) + np.sum(loggamma(GAMMA[i]))
        E_log_q_theta += -np.sum((GAMMA[i]-1) * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))

    elbo += E_log_q_theta

    ## Sixth term: -\sum_{i=1}^N \sum_{j=1}^M (E[log q(Z_ij)])
    E_log_q_z = 0
    for i in range(N):
        article = train_articles[i]
        nonzero_idx = train_nonzero_idxs[i]

        corr_idx = 0
        for idx in nonzero_idx:
            E_log_q_z += -article[idx] * np.sum(PHI[i][corr_idx] * np.log(PHI[i][corr_idx]))

            corr_idx += 1

        # Check if number of updates match with number of words
        assert(corr_idx == len(nonzero_idx))

    elbo += E_log_q_z

    print('ELBO: {}'.format(elbo))

    return elbo

# Runs CAVI for LDA
def run_cavi(LAMBDA, GAMMA, PHI, train_articles, train_nonzero_idxs, C, max_iter, tolerance=1e-4, ETA=0.01, ALPHA=0.1, predict_flag=False):
    # Unpack initial variational parameters
    LAMBDA_t = copy.deepcopy(LAMBDA) # Shape: (C,V)
    GAMMA_t = copy.deepcopy(GAMMA) # Shape: (N,C)
    PHI_t = copy.deepcopy(PHI) # Shape: (N,n_words,C)

    # Number of articles, vocabulary size
    N, V = train_articles.shape

    elbos = []
    times = []  # List to store time for each iteration
    
    # Initial time
    start_time = time.time()
    current_time = 0
    times.append(current_time)

    print('Running CAVI for LDA (C: {}, Iter: {}, Tolerance: {})...'.format(C, max_iter, tolerance))
    
    # Compute initial ELBO
    prev_elbo = compute_elbo(LAMBDA_t, GAMMA_t, PHI_t, train_articles, train_nonzero_idxs, C, ETA, ALPHA)
    elbos.append(prev_elbo)
    
    # Iteration counter
    iter_count = 0
    
    # Run for at most max_iter iterations
    while iter_count < max_iter:
        iter_count += 1
        iter_start_time = time.time()
        
        print('Iteration {}'.format(iter_count))
        print('Updating PHI and GAMMA')

        # For each document
        for i in tqdm(range(N)):
            article = train_articles[i]
            nonzero_idx = train_nonzero_idxs[i]

            # Fetch for PHI_ij update
            GAMMA_i_t = copy.deepcopy(GAMMA_t[i]) # C-vector

            # For each word in document
            corr_idx = 0

            # Iterate through each word with non-zero count on document
            for idx in nonzero_idx:
                log_PHI_ij = np.zeros((C,))

                for k in range(C):
                    # Fetch for PHI_ij update
                    LAMBDA_k_t = copy.deepcopy(LAMBDA_t[k]) # V-vector

                    exponent = digamma(GAMMA_i_t[k]) - digamma(np.sum(GAMMA_i_t))
                    exponent += digamma(LAMBDA_k_t[idx]) - digamma(np.sum(LAMBDA_k_t))
                    log_PHI_ij[k] = exponent

                # Normalize using log-sum-exp trick
                PHI_ij = np.exp(log_PHI_ij - compute_logsumexp(log_PHI_ij))
                try:
                    assert(np.abs(np.sum(PHI_ij) - 1) < 1e-6)
                except:
                    raise AssertionError('phi_ij: {}, Sum: {}'.format(PHI_ij, np.sum(PHI_ij)))

                PHI_t[i][corr_idx] = PHI_ij
                corr_idx += 1

            # Check if number of updates match with number of words
            assert(corr_idx == len(nonzero_idx))

            # Update GAMMA_i
            GAMMA_i_t = np.zeros((C,)) + ALPHA

            for k in range(C):
                GAMMA_i_t[k] += np.sum(article[nonzero_idx] * PHI_t[i][:,k])

            GAMMA_t[i] = GAMMA_i_t

        if not predict_flag:
            # For each topic
            print('Updating LAMBDA')

            for k in tqdm(range(C)):
                LAMBDA_k_t = np.zeros((V,)) + ETA

                # For each document
                for i in range(N):
                    article = train_articles[i]
                    nonzero_idx = train_nonzero_idxs[i]

                    # For each word in document
                    corr_idx = 0

                    for idx in nonzero_idx:
                        LAMBDA_k_t[idx] += article[idx] * PHI_t[i][corr_idx][k]
                        corr_idx +=1

                    # Check if number of updates match with number of words
                    assert(corr_idx == len(nonzero_idx))

                LAMBDA_t[k] = LAMBDA_k_t

        # Update time
        current_time = time.time() - start_time
        times.append(current_time)
        
        # Compute ELBO
        current_elbo = compute_elbo(LAMBDA_t, GAMMA_t, PHI_t, train_articles, train_nonzero_idxs, C, ETA, ALPHA)
        elbos.append(current_elbo)
        
        # Check for convergence
        elbo_change = np.abs(current_elbo - prev_elbo)
        print(f"ELBO change: {elbo_change}")
        
        if elbo_change < tolerance:
            print(f"Converged after {iter_count} iterations (ELBO change: {elbo_change} < tolerance: {tolerance})")
            break
            
        prev_elbo = current_elbo

    LAMBDA_final = copy.deepcopy(LAMBDA_t)
    GAMMA_final = copy.deepcopy(GAMMA_t)
    PHI_final = copy.deepcopy(PHI_t)

    return LAMBDA_final, GAMMA_final, PHI_final, elbos, times, iter_count

# Function to generate synthetic LDA data
def generate_lda_data(vocab_size, n_topics, n_docs, doc_length, alpha0=0.1, eta0=0.01, seed=42):
    """Generate synthetic LDA data"""
    np.random.seed(seed)
    
    # Generate true topics
    true_topics = np.zeros((n_topics, vocab_size))
    for k in range(n_topics):
        true_topics[k] = np.random.dirichlet(np.ones(vocab_size) * eta0)
    
    # Generate documents
    documents = np.zeros((n_docs, vocab_size), dtype=np.float32)
    true_doc_topics = np.zeros((n_docs, n_topics))
    
    # Track nonzero indices for each document
    nonzero_idxs = []
    
    for d in range(n_docs):
        # Generate document-topic proportions
        theta = np.random.dirichlet(np.ones(n_topics) * alpha0)
        true_doc_topics[d] = theta
        
        # Generate document
        doc_len = np.random.poisson(doc_length)
        doc_nonzero = []
        
        for _ in range(doc_len):
            # Sample topic
            z = np.random.choice(n_topics, p=theta)
            # Sample word
            w = np.random.choice(vocab_size, p=true_topics[z])
            # Add to document
            documents[d, w] += 1
            # Add to nonzero indices if not already there
            if w not in doc_nonzero:
                doc_nonzero.append(w)
        
        nonzero_idxs.append(sorted(doc_nonzero))
    
    return documents, nonzero_idxs, true_topics, true_doc_topics

# Main function to demonstrate the algorithm
def main():
    # Parameters
    vocab_size = 10000
    n_topics = 40
    n_docs = 2000
    doc_length = 30
    max_iter = 300
    tolerance = 0.1  # Convergence tolerance for ELBO change
    
    # Hyperparameters
    ETA = 0.01  # Topic Dirichlet prior
    ALPHA = 0.1  # Topic proportion Dirichlet prior
    
    # Generate synthetic data
    print("Generating synthetic LDA data...")
    documents, nonzero_idxs, true_topics, true_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, ALPHA, ETA
    )
    
    # Initialize variational parameters
    LAMBDA, GAMMA, PHI = init_var_param(documents, n_topics)
    
    # Run CAVI
    LAMBDA_final, GAMMA_final, PHI_final, elbos, times, iter_count = run_cavi(
        LAMBDA, GAMMA, PHI, documents, nonzero_idxs, n_topics, max_iter, tolerance, ETA, ALPHA
    )
    
    df = pd.DataFrame({"elbo": elbos, "time": times})
    df.to_csv("cavi_big_dataset.csv")
    # Plot ELBO vs time
    plt.figure(figsize=(12, 6))
    
    # Create a subplot for ELBO vs Time
    plt.plot(times, elbos)
    plt.xlabel('Time (seconds)')
    plt.ylabel('ELBO')
    plt.title('ELBO Convergence over Time')
    plt.grid(True)
    plt.show()
    
    # Print convergence information
    if iter_count < max_iter:
        print(f"Algorithm converged after {iter_count} iterations (tolerance: {tolerance})")
    else:
        print(f"Algorithm reached maximum iterations ({max_iter}) without convergence")

    return LAMBDA_final, GAMMA_final, PHI_final, elbos, times

if __name__ == "__main__":
    main()