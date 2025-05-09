import numpy as np
import pandas as pd
import time
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.optim import Adagrad
import matplotlib.pyplot as plt
from tqdm import tqdm

class LDARBCVBBVI(torch.nn.Module):
    def __init__(self, vocab_size, n_topics, alpha0=0.1, eta0=0.01):
        super(LDARBCVBBVI, self).__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.alpha0 = alpha0
        self.eta0 = eta0
        self.epsilon = 1e-10

        # Log-parameterization for unconstrained optimization
        # Initialize close to uniform distribution for stability
        self.lambda_topics = torch.nn.Parameter(
            torch.zeros(n_topics, vocab_size) + np.log(1.0 / vocab_size)
        )
        self.gamma = None
        self.n_docs = None

        # Cache for prior distributions
        self._topic_prior = None
        self._doc_prior = None

    def setup_doc_params(self, n_docs):
        self.n_docs = n_docs
        # Initialize close to uniform distribution for stability
        self.gamma = torch.nn.Parameter(
            torch.zeros(n_docs, self.n_topics) + np.log(1.0 / self.n_topics)
        )
        # Reset document prior for new batch
        self._doc_prior = None

    def get_var_params(self):
        # Apply softplus to ensure all values are positive
        lambda_params = F.softplus(self.lambda_topics) + self.epsilon
        gamma_params = F.softplus(self.gamma) + self.epsilon
        return lambda_params, gamma_params
    
    def get_topics(self):
        return torch.softmax(self.lambda_topics, dim=1)

    def topic_prior(self):
        # Cache the topic prior distribution
        if self._topic_prior is None:
            self._topic_prior = dist.Dirichlet(torch.ones(self.vocab_size) * self.eta0)
        return self._topic_prior

    def doc_prior(self):
        # Cache the document prior distribution
        if self._doc_prior is None:
            self._doc_prior = dist.Dirichlet(torch.ones(self.n_topics) * self.alpha0)
        return self._doc_prior

    def sample_variational(self, num_samples=1):
        """Fully vectorized sampling for higher speed"""
        lambda_params, gamma_params = self.get_var_params()
        
        # Single call to rsample with a batch dimension for topics
        topic_dist = dist.Dirichlet(lambda_params)
        topics = topic_dist.rsample((num_samples,))  # Shape: [num_samples, n_topics, vocab_size]
        
        # Single call to rsample with a batch dimension for doc topics
        doc_dist = dist.Dirichlet(gamma_params)
        doc_topics = doc_dist.rsample((num_samples,))  # Shape: [num_samples, n_docs, n_topics]
        
        if num_samples == 1:
            return topics.squeeze(0), doc_topics.squeeze(0)
        else:
            return topics, doc_topics

    def log_joint_prob(self, topics, doc_topics, docs):
        """Compute log joint probability with vectorized operations"""
        # Log prior for topics - vectorized
        topic_prior = self.topic_prior()
        
        # Handle different input shapes for topics
        if topics.dim() == 2:  # Single sample
            log_p_topics = topic_prior.log_prob(topics).sum()
        else:  # Multiple samples: [num_samples, n_topics, vocab_size]
            log_p_topics = torch.stack([topic_prior.log_prob(topics[i]).sum() for i in range(topics.shape[0])])
        
        # Log prior for topic proportions - vectorized
        doc_prior = self.doc_prior()
        
        # Handle different input shapes for doc_topics
        if doc_topics.dim() == 2:  # Single sample
            log_p_doc_topics = doc_prior.log_prob(doc_topics).sum()
        else:  # Multiple samples: [num_samples, n_docs, n_topics]
            log_p_doc_topics = torch.stack([doc_prior.log_prob(doc_topics[i]).sum() for i in range(doc_topics.shape[0])])
        
        # Log likelihood for documents - vectorized matrix multiplication
        # Shape: [n_docs, vocab_size] or [num_samples, n_docs, vocab_size]
        if doc_topics.dim() == 2 and topics.dim() == 2:  # Single sample
            word_probs = torch.matmul(doc_topics, topics)
            mask = docs > 0
            log_probs = torch.log(word_probs[mask] + self.epsilon)
            log_lik = torch.sum(docs[mask] * log_probs)
        else:  # Multiple samples
            log_lik = torch.zeros(topics.shape[0], device=topics.device)
            for i in range(topics.shape[0]):
                word_probs = torch.matmul(doc_topics[i], topics[i])
                mask = docs > 0
                log_probs = torch.log(word_probs[mask] + self.epsilon)
                log_lik[i] = torch.sum(docs[mask] * log_probs)
        
        if topics.dim() == 2:  # Single sample
            return log_p_topics + log_p_doc_topics + log_lik
        else:  # Multiple samples
            return log_p_topics + log_p_doc_topics + log_lik

    def log_q(self, topics, doc_topics):
        """Compute log of variational distribution with vectorized operations"""
        lambda_params, gamma_params = self.get_var_params()
        
        # Log q for topics - vectorized
        topic_dist = dist.Dirichlet(lambda_params)
        
        # Handle different input shapes
        if topics.dim() == 2:  # Single sample
            log_q_topics = topic_dist.log_prob(topics).sum()
        else:  # Multiple samples
            log_q_topics = torch.stack([topic_dist.log_prob(topics[i]).sum() for i in range(topics.shape[0])])
        
        # Log q for topic proportions - vectorized
        doc_dist = dist.Dirichlet(gamma_params)
        
        # Handle different input shapes
        if doc_topics.dim() == 2:  # Single sample
            log_q_doc_topics = doc_dist.log_prob(doc_topics).sum()
        else:  # Multiple samples
            log_q_doc_topics = torch.stack([doc_dist.log_prob(doc_topics[i]).sum() for i in range(doc_topics.shape[0])])
        
        return log_q_topics + log_q_doc_topics
    
    def elbo_loss(self, docs, n_samples=1):
        """Compute negative ELBO loss with multiple samples for variance reduction"""
        # Get samples in a vectorized manner
        topics, doc_topics = self.sample_variational(n_samples)
        
        # Compute log joint and log q in a batch for all samples
        log_joint = self.log_joint_prob(topics, doc_topics, docs)
        log_q_val = self.log_q(topics, doc_topics)
        
        # Compute ELBO
        elbo = log_joint - log_q_val
        
        # Return negative mean ELBO (we want to maximize ELBO or minimize -ELBO)
        return -torch.mean(elbo)
    
    def elbo_estimate(self, docs, n_samples=5):
        """Compute ELBO estimate with Monte Carlo samples"""
        with torch.no_grad():
            # Get samples in a vectorized manner
            topics, doc_topics = self.sample_variational(n_samples)
            
            # Compute log joint and log q in a batch for all samples
            log_joint = self.log_joint_prob(topics, doc_topics, docs)
            log_q_val = self.log_q(topics, doc_topics)
            
            # Compute and return mean ELBO
            elbo = torch.mean(log_joint - log_q_val)
            return elbo  # Return tensor, not item()

def generate_lda_data(vocab_size, n_topics, n_docs, doc_length, alpha0=0.1, eta0=0.01, seed=42):
    """Generate synthetic LDA data"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate true topics
    true_topics = np.zeros((n_topics, vocab_size))
    for k in range(n_topics):
        true_topics[k] = np.random.dirichlet(np.ones(vocab_size) * eta0)
    
    # Generate documents - vectorized where possible
    documents = np.zeros((n_docs, vocab_size), dtype=np.int32)
    true_doc_topics = np.zeros((n_docs, n_topics))
    
    for d in range(n_docs):
        # Draw topic proportions
        theta = np.random.dirichlet(np.ones(n_topics) * alpha0)
        true_doc_topics[d] = theta
        
        # Generate document
        doc_len = np.random.poisson(doc_length)
        topic_assignments = np.random.choice(n_topics, size=doc_len, p=theta)
        
        # Count words
        for z in topic_assignments:
            w = np.random.choice(vocab_size, p=true_topics[z])
            documents[d, w] += 1
    
    return documents, true_topics, true_doc_topics

def train_rbcv_bbvi(docs, n_topics, alpha0=0.1, eta0=0.01, n_iterations=600, 
                    initial_lr=0.1, n_samples=50, device='cpu',
                    scheduler_step_size=100, scheduler_gamma=0.5):
    """
    Train LDA model with RBCV-BBVI using PyTorch's autograd, AdaGrad, and learning rate scheduling
    """
    # Move data to specified device
    docs = docs.to(device)
    
    vocab_size = docs.shape[1]
    n_docs = docs.shape[0]
    
    # Initialize model and move to device
    model = LDARBCVBBVI(vocab_size, n_topics, alpha0, eta0).to(device)
    model.setup_doc_params(n_docs)
    
    # Initialize optimizers with AdaGrad
    topic_optimizer = Adagrad([model.lambda_topics], lr=initial_lr, weight_decay=1e-5)
    doc_optimizer = Adagrad([model.gamma], lr=initial_lr*2, weight_decay=1e-5)
    
    # Add learning rate schedulers
    topic_scheduler = torch.optim.lr_scheduler.StepLR(
        topic_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    doc_scheduler = torch.optim.lr_scheduler.StepLR(
        doc_optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
    
    # Initialize tracking variables
    elbo_values = []
    lr_values = []  # We'll keep this for AdaGrad's effective rate
    
    # Training loop
    for iteration in tqdm(range(n_iterations)):
        # Multiple updates for document parameters
        for _ in range(2):
            doc_optimizer.zero_grad()
            loss = model.elbo_loss(docs, n_samples=n_samples//2)
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.gamma, 5.0)
            doc_optimizer.step()
        
        # Update topic parameters
        topic_optimizer.zero_grad()
        loss = model.elbo_loss(docs, n_samples=n_samples)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.lambda_topics, 5.0)
        topic_optimizer.step()
        
        # Step the learning rate schedulers
        topic_scheduler.step()
        doc_scheduler.step()
        
        # Compute and store ELBO at current iteration
        with torch.no_grad():
            elbo_val = model.elbo_estimate(docs, n_samples=5).item()
            elbo_values.append((iteration, elbo_val))
        
        # Track AdaGrad effective learning rates (approximation using first parameter)
        if hasattr(topic_optimizer, 'param_groups') and len(topic_optimizer.param_groups) > 0:
            param_state = list(topic_optimizer.state.values())[0] if topic_optimizer.state else {}
            if 'sum' in param_state:
                effective_lr = initial_lr / (torch.sqrt(param_state['sum'][0, 0]) + topic_optimizer.defaults['eps'])
                lr_values.append((iteration, effective_lr.item()))
        
        # Print progress
        if (iteration + 1) % 25 == 0:
            current_elbo = elbo_values[-1][1] if elbo_values else 0
            print(f"Iteration {iteration+1}/{n_iterations}, ELBO: {current_elbo:.2f}")
    
    return model, elbo_values, lr_values


def main():
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parameters
    vocab_size = 500
    n_topics = 5
    n_docs = 50
    doc_length = 100
    alpha0 = 0.1
    eta0 = 0.01
    n_iterations = 10000
    initial_lr = 0.09
    n_samples = 30
    
    # Scheduler parameters
    scheduler_step_size = 200  # Number of iterations before reducing learning rate
    scheduler_gamma = 0.5      # Factor by which to reduce learning rate
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic LDA data...")
    documents, true_topics, true_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, alpha0, eta0
    )
    docs = torch.tensor(documents, dtype=torch.float32)
    
    # Train model with RBCV-BBVI using AdaGrad and learning rate scheduling
    print("Training LDA model with RBCV-BBVI using AdaGrad and learning rate scheduling...")
    start = time.time()
    model, elbo_values, lr_values = train_rbcv_bbvi(
        docs, n_topics, alpha0, eta0, n_iterations, initial_lr, n_samples, device,
        scheduler_step_size, scheduler_gamma
    )
    stop = time.time()
    
    # Extract iteration and ELBO values for plotting
    iterations, elbos = zip(*elbo_values)
    time_steps = np.linspace(0, stop-start, len(elbos))
    df = pd.DataFrame({"elbo": elbos, "time_steps": time_steps})
    df.to_csv("rbcv_bbvi_dataset.csv")
    
    # Apply moving average to smooth the plot
    window_size = 10
    if len(elbos) > window_size:
        elbo_smoothed = np.convolve(elbos, np.ones(window_size)/window_size, mode='valid')
        smooth_iterations = iterations[window_size-1:len(iterations)]
    else:
        elbo_smoothed = elbos
        smooth_iterations = iterations
    
    # Plot ELBO trace
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, elbos, 'o-', alpha=0.4, label='Raw ELBO')
    plt.plot(smooth_iterations, elbo_smoothed, 'r-', linewidth=2, label='Smoothed ELBO')
    plt.title('ELBO Convergence with RBCV-BBVI using AdaGrad + Scheduler')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.grid(True)
    plt.legend()
    # plt.savefig('rbcv_bbvi_elbo_trace.png')
    plt.show()
    
    # Plot AdaGrad effective learning rate if values were tracked
    if lr_values:
        lr_iterations, lr_rates = zip(*lr_values)
        plt.figure(figsize=(10, 4))
        plt.plot(lr_iterations, lr_rates)
        plt.title('AdaGrad Effective Learning Rate with Scheduler')
        plt.xlabel('Iteration')
        plt.ylabel('Effective Learning Rate')
        plt.grid(True)
        plt.yscale('log')
        # plt.savefig('adagrad_lr_trace.png')
        plt.show()
        print("AdaGrad learning rate plot saved as 'adagrad_lr_trace.png'")
    
    print("ELBO trace plot saved as 'rbcv_bbvi_elbo_trace.png'")

if __name__ == "__main__":
    main()