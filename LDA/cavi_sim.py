import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, loggamma
from scipy.sparse import csr_matrix
from tqdm import tqdm
import time
import copy


class LDADataGenerator:
    """Simple class to generate synthetic LDA data"""
    def __init__(self, seed=None):
        self.random_state = np.random.RandomState(seed)
    
    def simulate_LDA(self, N, Ms, K, V, eta0, alpha0):
        """Simulate data from LDA model"""
        # Generate true parameters
        beta = self.random_state.dirichlet(np.full(V, eta0), size=K)
        theta = self.random_state.dirichlet(np.full(K, alpha0), size=N)
        
        row_idxs = []
        col_idxs = []
        values = []
        nonzero_idxs = []

        # Generate documents
        for i in range(N):
            doc_word_counts = np.zeros(V)
            for _ in range(Ms[i]):
                z_ij = self.random_state.choice(K, p=theta[i])
                x_ij = self.random_state.choice(V, p=beta[z_ij])
                doc_word_counts[x_ij] += 1
            doc_nonzero = np.nonzero(doc_word_counts)[0]
            doc_nonzero = np.array(sorted(doc_nonzero))
            nonzero_idxs.append(doc_nonzero)

            row_idxs.extend([i] * len(doc_nonzero))
            col_idxs.extend(doc_nonzero)
            values.extend(doc_word_counts[doc_nonzero])
        
        # Convert to dense matrix
        documents = csr_matrix((values, (row_idxs, col_idxs)), shape=(N, V)).toarray()
        
        return documents, nonzero_idxs, beta, theta


class LDA:
    """Latent Dirichlet Allocation with CAVI"""
    def __init__(self, K, eta0, alpha0, seed=None):
        self.K = K
        self.eta0 = eta0
        self.alpha0 = alpha0
        
        if seed is not None:
            np.random.seed(seed)
    
    def log_sum_exp(self, vec):
        """Compute log(sum(exp(vec))) in a numerically stable way"""
        vec_max = np.max(vec, axis=0)
        exp_vec = np.exp(vec - vec_max)
        sum_exp_vec = np.sum(exp_vec)
        log_sum_exp = np.log(sum_exp_vec) + vec_max
        return log_sum_exp
    
    def init_variational_params(self, documents):
        """Initialize variational parameters"""
        N, V = documents.shape
        
        # Lambda (topics)
        lambda_ = np.random.uniform(low=0.01, high=1.0, size=(self.K, V))
        # lambda_ = np.random.uniform(low=0.01, high=1.0, size=(self.K, V)) * 0.01 + np.log(1.0 / V)

        # Gamma (topic proportions)
        gamma = np.ones((N, self.K))
        
        # Phi (topic assignments)
        phi = []
        for document in documents:
            M = np.sum((document > 0).astype("int32"))
            doc_phi = np.ones((M, self.K)) / self.K
            phi.append(doc_phi)
            
        return lambda_, gamma, phi
    
    def compute_ELBO(self, lambda_, gamma, phi, documents, nonzero_idxs):
        """
        Compute Evidence Lower Bound (ELBO) following the exact implementation
        in the attached document.
        """
        N, V = documents.shape
        
        # 1. Expected log prior of topics: E[log p(beta)]
        E_log_p_beta = 0
        for k in range(self.K):
            E_log_p_beta += (self.eta0-1) * np.sum(digamma(lambda_[k]) - digamma(np.sum(lambda_[k])))
        
        # 2. Expected log prior of topic proportions: E[log p(theta)]
        E_log_p_theta = 0
        for i in range(N):
            E_log_p_theta += (self.alpha0-1) * np.sum(digamma(gamma[i]) - digamma(np.sum(gamma[i])))
        
        # 3. Expected log likelihood: E[log p(x,z|theta,beta)]
        E_log_p_x_z = 0
        for i in range(N):
            document = documents[i]
            nonzero_idx = nonzero_idxs[i]
            
            word_idx = 0
            for idx in nonzero_idx:
                # Expected log probability of topic assignments given proportions
                E_log_p_x_z += document[idx] * np.sum(phi[i][word_idx] * (digamma(gamma[i]) - digamma(np.sum(gamma[i]))))
                
                # Expected log probability of words given topic assignments
                E_log_p_x_z += document[idx] * np.sum(phi[i][word_idx] * (digamma(lambda_[:, idx]) - digamma(np.sum(lambda_, axis=1))))
                
                word_idx += 1
        
        # 4. Negative entropy of variational topics (q(beta)): -H[q(beta)]
        E_log_q_beta = 0
        for k in range(self.K):
            E_log_q_beta += -loggamma(np.sum(lambda_[k])) + np.sum(loggamma(lambda_[k]))
            E_log_q_beta += -np.sum((lambda_[k]-1) * (digamma(lambda_[k]) - digamma(np.sum(lambda_[k]))))
        
        # 5. Negative entropy of variational topic proportions (q(theta)): -H[q(theta)]
        E_log_q_theta = 0
        for i in range(N):
            E_log_q_theta += -loggamma(np.sum(gamma[i])) + np.sum(loggamma(gamma[i]))
            E_log_q_theta += -np.sum((gamma[i]-1) * (digamma(gamma[i]) - digamma(np.sum(gamma[i]))))
        
        # 6. Negative entropy of variational topic assignments (q(z)): -H[q(z)]
        E_log_q_z = 0
        for i in range(N):
            document = documents[i]
            nonzero_idx = nonzero_idxs[i]
            
            word_idx = 0
            for idx in nonzero_idx:
                # For numerical stability, only consider non-zero probabilities
                phi_ij = phi[i][word_idx]
                mask = phi_ij > 1e-10
                if np.any(mask):
                    E_log_q_z += document[idx] * np.sum(phi_ij[mask] * np.log(phi_ij[mask]))
                
                word_idx += 1
        
        # Combine all terms following the original implementation
        # ELBO = E[log p(beta)] + E[log p(theta)] + E[log p(x,z|theta,beta)] - E[log q(beta)] - E[log q(theta)] - E[log q(z)]
        # Note: The key difference is that we SUBTRACT the entropies rather than adding them
        elbo = E_log_p_beta + E_log_p_theta + E_log_p_x_z + E_log_q_beta + E_log_q_theta + E_log_q_z
        
        return elbo
    
    def fit(self, documents, nonzero_idxs=None, max_iterations=100, tol=1.0, verbose=True):
        """Fit LDA model using CAVI"""
        self.documents = documents
        self.N, self.V = documents.shape
        
        # Create nonzero indices if not provided
        if nonzero_idxs is None:
            self.nonzero_idxs = []
            for i in range(self.N):
                doc_nonzero = np.nonzero(self.documents[i])[0]
                self.nonzero_idxs.append(np.array(sorted(doc_nonzero)))
        else:
            self.nonzero_idxs = nonzero_idxs
        
        # Initialize variational parameters
        self.lambda_, self.gamma, self.phi = self.init_variational_params(documents)
        
        # Run CAVI
        self._run_cavi(max_iterations, tol, verbose)
        
        return self
    
    def _run_cavi(self, max_iterations, tol, verbose):
        """Run Coordinate Ascent Variational Inference"""
        # Make copies of variational parameters
        lambda_t = copy.deepcopy(self.lambda_)
        gamma_t = copy.deepcopy(self.gamma)
        phi_t = copy.deepcopy(self.phi)
        
        self.ELBOs = []
        start = time.time()
        
        # Initial ELBO
        curr_ELBO = self.compute_ELBO(lambda_t, gamma_t, phi_t, self.documents, self.nonzero_idxs)
        self.ELBOs.append(curr_ELBO)
        
        if verbose:
            print(f"Initial ELBO: {curr_ELBO}")
        
        # CAVI iterations
        for t in range(max_iterations):
            if verbose:
                print(f"Iteration {t+1}")
            
            # Update phi and gamma
            for i in tqdm(range(self.N), desc="Updating phi and gamma", disable=not verbose):
                document = self.documents[i]
                nonzero_idx = self.nonzero_idxs[i]
                gamma_i_t = np.zeros(self.K) + self.alpha0
                
                word_idx = 0
                for idx in nonzero_idx:
                    # Compute log phi
                    log_phi_ij = np.zeros(self.K)
                    for k in range(self.K):
                        log_phi_ij[k] = digamma(gamma_t[i, k]) - digamma(np.sum(gamma_t[i]))
                        log_phi_ij[k] += digamma(lambda_t[k, idx]) - digamma(np.sum(lambda_t[k]))
                    
                    # Normalize using log-sum-exp trick
                    log_norm = self.log_sum_exp(log_phi_ij)
                    phi_ij = np.exp(log_phi_ij - log_norm)
                    phi_t[i][word_idx] = phi_ij
                    
                    # Update gamma
                    gamma_i_t += document[idx] * phi_ij
                    
                    word_idx += 1
                
                gamma_t[i] = gamma_i_t
            
            # Update lambda
            for k in tqdm(range(self.K), desc="Updating lambda", disable=not verbose):
                lambda_k_t = np.zeros(self.V) + self.eta0
                
                for i in range(self.N):
                    document = self.documents[i]
                    nonzero_idx = self.nonzero_idxs[i]
                    
                    word_idx = 0
                    for idx in nonzero_idx:
                        lambda_k_t[idx] += document[idx] * phi_t[i][word_idx][k]
                        word_idx += 1
                
                lambda_t[k] = lambda_k_t
            
            # Compute ELBO
            prev_ELBO = curr_ELBO
            curr_ELBO = self.compute_ELBO(lambda_t, gamma_t, phi_t, self.documents, self.nonzero_idxs)
            self.ELBOs.append(curr_ELBO)
            
            if verbose:
                print(f"Current ELBO: {curr_ELBO} | Change in ELBO: {curr_ELBO - prev_ELBO}\n")
            
            # Check convergence
            if abs(curr_ELBO - prev_ELBO) < tol:
                if verbose:
                    print(f"Converged after {t+1} iterations")
                break
        
        # Store final parameters
        self.lambda_ = copy.deepcopy(lambda_t)
        self.gamma = copy.deepcopy(gamma_t)
        self.phi = copy.deepcopy(phi_t)
        
        stop = time.time()
        self.fit_time = stop - start
        
        if verbose:
            print(f"Fitting took {self.fit_time:.2f} seconds")
    
    def plot_ELBO(self):
        """Plot ELBO convergence"""
        plt.figure(figsize=(10, 6))
        plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        plt.plot(self.ELBOs[1:])
        plt.xlabel('Iteration')
        plt.ylabel('ELBO')
        plt.title('ELBO Convergence')
        plt.grid(True)
        plt.show()
    
    def compute_test_likelihood(self, test_documents, test_nonzero_idxs=None, max_iter=20):
        """
        Compute log likelihood on test documents
        
        Parameters:
        -----------
        test_documents: array-like, shape (n_docs, V), test document-term matrix
        test_nonzero_idxs: list of arrays, indices of non-zero elements in each test document
        max_iter: int, maximum number of iterations for inference
        
        Returns:
        --------
        log_likelihood: float, log likelihood of test documents
        """
        n_docs, V = test_documents.shape
        
        # Create nonzero indices if not provided
        if test_nonzero_idxs is None:
            test_nonzero_idxs = []
            for i in range(n_docs):
                doc_nonzero = np.nonzero(test_documents[i])[0]
                test_nonzero_idxs.append(np.array(sorted(doc_nonzero)))
        
        # Initialize variational parameters for test documents
        _, gamma, phi = self.init_variational_params(test_documents)
        
        # Run inference on test documents
        for _ in range(max_iter):
            # Update phi and gamma
            for i in range(n_docs):
                document = test_documents[i]
                nonzero_idx = test_nonzero_idxs[i]
                gamma_i = np.zeros(self.K) + self.alpha0
                
                word_idx = 0
                for idx in nonzero_idx:
                    # Compute log phi
                    log_phi_ij = np.zeros(self.K)
                    for k in range(self.K):
                        log_phi_ij[k] = digamma(gamma[i, k]) - digamma(np.sum(gamma[i]))
                        log_phi_ij[k] += digamma(self.lambda_[k, idx]) - digamma(np.sum(self.lambda_[k]))
                    
                    # Normalize
                    log_norm = self.log_sum_exp(log_phi_ij)
                    phi_ij = np.exp(log_phi_ij - log_norm)
                    phi[i][word_idx] = phi_ij
                    
                    # Update gamma
                    gamma_i += document[idx] * phi_ij
                    
                    word_idx += 1
                
                gamma[i] = gamma_i
        
        # Compute log likelihood using the variational bound
        log_likelihood = 0
        
        for i in range(n_docs):
            # Compute document-specific log likelihood
            doc_log_likelihood = 0
            document = test_documents[i]
            nonzero_idx = test_nonzero_idxs[i]
            
            # Expected log prior of topic proportions
            doc_log_likelihood += (self.alpha0-1) * np.sum(digamma(gamma[i]) - digamma(np.sum(gamma[i])))
            
            # Expected log likelihood
            word_idx = 0
            for idx in nonzero_idx:
                doc_log_likelihood += document[idx] * np.sum(phi[i][word_idx] * (digamma(gamma[i]) - digamma(np.sum(gamma[i]))))
                doc_log_likelihood += document[idx] * np.sum(phi[i][word_idx] * (digamma(self.lambda_[:, idx]) - digamma(np.sum(self.lambda_, axis=1))))
                word_idx += 1
            
            # Entropy of variational topic proportions
            doc_log_likelihood += -loggamma(np.sum(gamma[i])) + np.sum(loggamma(gamma[i]))
            doc_log_likelihood += -np.sum((gamma[i]-1) * (digamma(gamma[i]) - digamma(np.sum(gamma[i]))))
            
            # Entropy of variational topic assignments
            word_idx = 0
            for idx in nonzero_idx:
                mask = phi[i][word_idx] > 1e-10
                if np.any(mask):
                    doc_log_likelihood += -document[idx] * np.sum(phi[i][word_idx][mask] * np.log(phi[i][word_idx][mask]))
                word_idx += 1
            
            log_likelihood += doc_log_likelihood
        
        return log_likelihood


# Simple usage example
if __name__ == "__main__":
    # Generate synthetic data
    N = 1600       # Number of documents
    N_test = 400   # Number of test documents
    V = 10000      # Vocabulary size
    K = 60       # Number of topics
    alpha0 = 0.1  # Symmetric Dirichlet parameter for topic proportions
    eta0 = 0.01   # Symmetric Dirichlet parameter for topics
    
    # Generate document lengths
    Ms = np.random.poisson(300, size=N + N_test)
    
    # Generate data
    i = 100
    generator = LDADataGenerator(i)
    documents, nonzero_idxs, true_beta, true_theta = generator.simulate_LDA(
        N + N_test, Ms, K, V, eta0, alpha0
    )
    
    # Split into train and test
    train_documents = documents[:N]
    train_nonzero_idxs = nonzero_idxs[:N]
    test_documents = documents[N:]
    test_nonzero_idxs = nonzero_idxs[N:]
    
    # Fit LDA model
    lda = LDA(K=K, eta0=eta0, alpha0=alpha0, seed=i)
    lda.fit(train_documents, train_nonzero_idxs, max_iterations=1000, tol=10, verbose=True)
    
    # Plot ELBO
    lda.plot_ELBO()
    
    # Compute test likelihood
    test_likelihood = lda.compute_test_likelihood(test_documents, test_nonzero_idxs)
    print(f"Test log likelihood: {test_likelihood:.2f}")