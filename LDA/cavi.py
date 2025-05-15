import os
import numpy as np
import time
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.special import digamma, gammaln
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LDA_CAVI:
    def __init__(self, K, V, eta=0.01, alpha=0.1):
        """
        Initialize the LDA model for CAVI.
        
        Args:
            K: Number of topics
            V: Size of vocabulary
            eta: Dirichlet prior for topics (default: 0.01)
            alpha: Dirichlet prior for topic proportions (default: 0.1)
        """
        self.K = K
        self.V = V
        self.eta = eta
        self.alpha = alpha
        
    def generate_synthetic_data(self, N, doc_length_pois_lambda=70):
        """
        Generate synthetic LDA data.
        
        Args:
            N: Number of documents to generate
            doc_length_pois_lambda: Poisson parameter for document length
            
        Returns:
            documents: List of documents, each a list of word indices
            true_topics: The true topics used to generate the data
        """
        print("Generating synthetic LDA data...")
        
        # Generate true topics from Dirichlet
        true_topics = torch.distributions.Dirichlet(torch.ones(self.V) * self.eta).sample((self.K,))
        
        # Generate documents
        documents = []
        document_lengths = np.random.poisson(doc_length_pois_lambda, size=N)
        
        # Generate non-zero indices for tracking
        nonzero_idxs = []
        
        for i in range(N):
            Mi = document_lengths[i]
            
            # Generate topic proportions for this document
            theta = torch.distributions.Dirichlet(torch.ones(self.K) * self.alpha).sample()
            
            # Generate document
            doc = torch.zeros(self.V)
            nonzero_idx = []
            
            for j in range(Mi):
                # Sample topic
                z = torch.multinomial(theta, 1).item()
                
                # Sample word from topic
                w = torch.multinomial(true_topics[z], 1).item()
                
                # Add to document (bag of words representation)
                doc[w] += 1
                
                # Keep track of non-zero indices
                if w not in nonzero_idx:
                    nonzero_idx.append(w)
            
            documents.append(doc)
            nonzero_idxs.append(sorted(nonzero_idx))
        
        # Convert to tensors
        documents = torch.stack(documents)
        
        return documents, nonzero_idxs, true_topics
    
    def init_var_params(self, documents):
        """
        Initialize variational parameters.
        
        Args:
            documents: Tensor of document word counts, shape (N, V)
            
        Returns:
            lambda_: Variational parameters for topics, shape (K, V)
            gamma: Variational parameters for topic proportions, shape (N, K)
            phi: Variational parameters for topic assignments, list of N tensors of shape (n_words, K)
        """
        print("Initializing variational parameters...")
        
        N = documents.shape[0]
        
        # Initialize lambda (topics)
        lambda_ = torch.rand(self.K, self.V) + 0.01
        
        # Initialize gamma (topic proportions)
        gamma = torch.ones(N, self.K)
        
        # Initialize phi (topic assignments)
        phi = []
        for i in range(N):
            # Get number of words in document
            doc = documents[i]
            nonzero_idx = torch.nonzero(doc).squeeze()
            n_words = nonzero_idx.shape[0]
            
            # Initialize phi for this document
            doc_phi = torch.ones(n_words, self.K) / self.K
            phi.append(doc_phi)
        
        return lambda_, gamma, phi
    
    def compute_expected_log_probs(self, lambda_, gamma):
        """
        Compute expected log probabilities for topics and topic proportions.
        
        Args:
            lambda_: Topic variational parameters, shape (K, V)
            gamma: Topic proportion variational parameters, shape (N, K)
            
        Returns:
            E_log_beta: Expected log probability of words given topics, shape (K, V)
            E_log_theta: Expected log probability of topics given documents, shape (N, K)
        """
        # Compute E[log beta_kv]
        lambda_sum = torch.sum(lambda_, dim=1, keepdim=True)
        E_log_beta = digamma(lambda_.numpy()) - digamma(lambda_sum.numpy())
        E_log_beta = torch.from_numpy(E_log_beta)
        
        # Compute E[log theta_ik]
        gamma_sum = torch.sum(gamma, dim=1, keepdim=True)
        E_log_theta = digamma(gamma.numpy()) - digamma(gamma_sum.numpy())
        E_log_theta = torch.from_numpy(E_log_theta)
        
        return E_log_beta, E_log_theta
    
    def update_phi(self, documents, E_log_theta, E_log_beta, nonzero_idxs):
        """
        Update phi (topic assignments) variational parameters.
        
        Args:
            documents: Document word counts, shape (N, V)
            E_log_theta: Expected log probability of topics given documents, shape (N, K)
            E_log_beta: Expected log probability of words given topics, shape (K, V)
            nonzero_idxs: List of lists containing non-zero indices for each document
            
        Returns:
            phi: Updated variational parameters for topic assignments
        """
        N = documents.shape[0]
        phi = []
        
        for i in range(N):
            doc = documents[i]
            nonzero_idx = nonzero_idxs[i]
            
            # Initialize phi for this document
            doc_phi = torch.zeros(len(nonzero_idx), self.K)
            
            # Update phi for each word
            for j, v in enumerate(nonzero_idx):
                # Compute log phi
                log_phi = E_log_theta[i] + E_log_beta[:, v]
                
                # Normalize with log-sum-exp trick
                max_log_phi = torch.max(log_phi)
                log_phi_norm = log_phi - max_log_phi - torch.log(torch.sum(torch.exp(log_phi - max_log_phi)))
                
                # Convert to probabilities
                doc_phi[j] = torch.exp(log_phi_norm)
                
                # Check if normalized properly (prevent numerical issues)
                if not torch.isclose(torch.sum(doc_phi[j]), torch.tensor(1.0), atol=1e-3):
                    doc_phi[j] = doc_phi[j] / torch.sum(doc_phi[j])
            
            phi.append(doc_phi)
        
        return phi
    
    def update_gamma(self, documents, phi, nonzero_idxs):
        """
        Update gamma (topic proportions) variational parameters.
        
        Args:
            documents: Document word counts, shape (N, V)
            phi: Variational parameters for topic assignments
            nonzero_idxs: List of lists containing non-zero indices for each document
            
        Returns:
            gamma: Updated variational parameters for topic proportions
        """
        N = documents.shape[0]
        gamma = torch.zeros(N, self.K)
        
        for i in range(N):
            doc = documents[i]
            nonzero_idx = nonzero_idxs[i]
            doc_phi = phi[i]
            
            # Initialize gamma with alpha
            gamma[i] = self.alpha
            
            # Add contribution from each word
            for j, v in enumerate(nonzero_idx):
                word_count = doc[v].item()
                gamma[i] += word_count * doc_phi[j]
        
        return gamma
    
    def update_lambda(self, documents, phi, nonzero_idxs):
        """
        Update lambda (topics) variational parameters.
        
        Args:
            documents: Document word counts, shape (N, V)
            phi: Variational parameters for topic assignments
            nonzero_idxs: List of lists containing non-zero indices for each document
            
        Returns:
            lambda_: Updated variational parameters for topics
        """
        lambda_ = torch.ones(self.K, self.V) * self.eta
        
        for i in range(len(documents)):
            doc = documents[i]
            nonzero_idx = nonzero_idxs[i]
            doc_phi = phi[i]
            
            for j, v in enumerate(nonzero_idx):
                word_count = doc[v].item()
                lambda_[:, v] += word_count * doc_phi[j]
        
        return lambda_
    
    def compute_elbo(self, documents, lambda_, gamma, phi, nonzero_idxs):
        """
        Compute the ELBO.
        
        Args:
            documents: Document word counts, shape (N, V)
            lambda_: Variational parameters for topics
            gamma: Variational parameters for topic proportions
            phi: Variational parameters for topic assignments
            nonzero_idxs: List of lists containing non-zero indices for each document
            
        Returns:
            elbo: The evidence lower bound
        """
        # Compute expected log probs
        E_log_beta, E_log_theta = self.compute_expected_log_probs(lambda_, gamma)
        
        # Initialize ELBO components
        E_log_p_beta = 0
        E_log_p_theta = 0
        E_log_p_z_and_w = 0
        E_log_q_beta = 0
        E_log_q_theta = 0
        E_log_q_z = 0
        
        # E[log p(beta)] - Topics prior
        lambda_sum = torch.sum(lambda_, dim=1)
        E_log_p_beta = self.K * (torch.sum(torch.lgamma(torch.tensor(self.V * self.eta))) - 
                              self.V * torch.sum(torch.lgamma(torch.tensor(self.eta))))
        for k in range(self.K):
            E_log_p_beta += torch.sum((self.eta - 1) * E_log_beta[k])
        
        # E[log p(theta)] - Topic proportions prior
        gamma_sum = torch.sum(gamma, dim=1)
        E_log_p_theta = torch.sum(torch.lgamma(torch.tensor(self.K * self.alpha))) - \
                       self.K * torch.sum(torch.lgamma(torch.tensor(self.alpha)))
        for i in range(len(documents)):
            E_log_p_theta += torch.sum((self.alpha - 1) * E_log_theta[i])
        
        # E[log p(z, w|theta, beta)] - Joint likelihood of topic assignments and words
        for i in range(len(documents)):
            doc = documents[i]
            nonzero_idx = nonzero_idxs[i]
            doc_phi = phi[i]
            
            for j, v in enumerate(nonzero_idx):
                word_count = doc[v].item()
                for k in range(self.K):
                    E_log_p_z_and_w += word_count * doc_phi[j, k] * (E_log_theta[i, k] + E_log_beta[k, v])
        
        # Entropy terms (negative KL divergence)
        
        # -E[log q(beta)] - Topics variational posterior entropy
        for k in range(self.K):
            E_log_q_beta += -torch.lgamma(lambda_sum[k]) + torch.sum(torch.lgamma(lambda_[k]))
            E_log_q_beta += -torch.sum((lambda_[k] - 1) * E_log_beta[k])
        
        # -E[log q(theta)] - Topic proportions variational posterior entropy
        for i in range(len(documents)):
            E_log_q_theta += -torch.lgamma(gamma_sum[i]) + torch.sum(torch.lgamma(gamma[i]))
            E_log_q_theta += -torch.sum((gamma[i] - 1) * E_log_theta[i])
        
        # -E[log q(z)] - Topic assignments variational posterior entropy
        for i in range(len(documents)):
            doc = documents[i]
            nonzero_idx = nonzero_idxs[i]
            doc_phi = phi[i]
            
            for j, v in enumerate(nonzero_idx):
                word_count = doc[v].item()
                
                # Avoid NaN in log by adding small epsilon
                safe_phi = doc_phi[j] + 1e-10
                safe_phi = safe_phi / torch.sum(safe_phi)
                E_log_q_z += -word_count * torch.sum(safe_phi * torch.log(safe_phi))
        
        elbo = E_log_p_beta + E_log_p_theta + E_log_p_z_and_w + E_log_q_beta + E_log_q_theta + E_log_q_z
        
        return elbo.item()
    
    def fit(self, documents, nonzero_idxs, max_iter=100, convergence_threshold=1e-5):
        """
        Fit the CAVI algorithm to the data.
        
        Args:
            documents: Document word counts, shape (N, V)
            nonzero_idxs: List of lists containing non-zero indices for each document
            max_iter: Maximum number of iterations
            convergence_threshold: Convergence threshold for ELBO change
            
        Returns:
            lambda_: Final topics variational parameters
            gamma: Final topic proportions variational parameters
            elbo_values: List of ELBO values at each iteration
        """
        print(f"Fitting CAVI for LDA (max_iter={max_iter})...")
        
        # Initialize variational parameters
        lambda_, gamma, phi = self.init_var_params(documents)
        
        # Initialize ELBO tracking
        elbo_values = []
        prev_elbo = -float('inf')
        
        # CAVI iterations
        for iteration in tqdm(range(max_iter)):
            # Compute expected log probs
            E_log_beta, E_log_theta = self.compute_expected_log_probs(lambda_, gamma)
            
            # Update phi (topic assignments)
            phi = self.update_phi(documents, E_log_theta, E_log_beta, nonzero_idxs)
            
            # Update gamma (topic proportions)
            gamma = self.update_gamma(documents, phi, nonzero_idxs)
            
            # Update lambda (topics)
            lambda_ = self.update_lambda(documents, phi, nonzero_idxs)
            
            # Compute ELBO
            elbo = self.compute_elbo(documents, lambda_, gamma, phi, nonzero_idxs)
            elbo_values.append(elbo)
            
            # Check convergence
            elbo_change = abs(elbo - prev_elbo)
            if iteration > 0 and elbo_change < convergence_threshold:
                print(f"Converged at iteration {iteration} with ELBO change {elbo_change:.6f}")
                break
                
            prev_elbo = elbo
            
            # Print progress
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"Iteration {iteration+1}: ELBO = {elbo:.4f}")
        
        return lambda_, gamma, elbo_values

    def compute_predictive_likelihood(self, test_documents, test_nonzero_idxs, lambda_, gamma, num_samples=20):
        print("Computing predictive likelihood on test documents...")
        N_test, V = test_documents.shape
        K = lambda_.shape[0]
        hat_beta = lambda_ / torch.sum(lambda_, dim=1, keepdims=True)
        hat_theta = gamma / torch.sum(gamma, dim=1, keepdims=True)

        log_test_lik = 0
        for i in range(N_test):
            doc = test_documents[i]
            nonzero_idx = test_nonzero_idxs[i]
            for idx in nonzero_idx:
                log_test_lik += torch.log(torch.dot(hat_theta[i], hat_beta[:, idx]))

        return log_test_lik.item()

def main():
    # Set parameters
    K = 5  # Number of topics
    V = 1000  # Vocabulary size
    N = 200  # Number of documents
    N_test = 40  # Number of test documents
    max_iter = 100
    
    # Create LDA CAVI model
    lda_cavi = LDA_CAVI(K=K, V=V, eta=0.01, alpha=0.1)
    
    # Generate synthetic data
    all_documents, all_nonzero_idxs, true_topics = lda_cavi.generate_synthetic_data(N=N+N_test)
    
    # Split into train and test
    train_documents = all_documents[:N]
    train_nonzero_idxs = all_nonzero_idxs[:N]
    test_documents = all_documents[N:]
    test_nonzero_idxs = all_nonzero_idxs[N:]
    
    print(f"Training on {N} documents, testing on {N_test} documents")
    
    # Fit the model on training data
    lambda_, gamma, elbo_values = lda_cavi.fit(train_documents, train_nonzero_idxs, max_iter=max_iter)
    
    # Compute predictive likelihood on test documents
    pred_ll = lda_cavi.compute_predictive_likelihood(test_documents, test_nonzero_idxs, lambda_, gamma)

if __name__ == "__main__":
    main()