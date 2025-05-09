import numpy as np
import time
import pandas as pd
import torch
import torch.distributions as dist
import torch.nn.functional as F
from torch.optim import Adagrad
import matplotlib.pyplot as plt
from tqdm import tqdm

class LDABBVI(torch.nn.Module):
    """
    LDA model with vanilla Black-Box Variational Inference implementation
    """
    def __init__(self, vocab_size, n_topics, alpha0=0.1, eta0=0.01):
        super(LDABBVI, self).__init__()
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
        """Sample from the variational distributions"""
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
        """Compute log joint probability"""
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
        """Compute log of variational distribution"""
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
    
    def topic_score(self, topics):
        """Compute score function for topics variational parameters"""
        lambda_params, _ = self.get_var_params()
        
        # Get digamma terms
        lambda_sum = lambda_params.sum(dim=1, keepdim=True)
        digamma_diff = torch.digamma(lambda_sum) - torch.digamma(lambda_params)
        
        # Score is digamma difference plus log of sample
        score = digamma_diff + torch.log(topics + self.epsilon)
        
        return score
    
    def doc_score(self, doc_topics):
        """Compute score function for document topic proportions variational parameters"""
        _, gamma_params = self.get_var_params()
        
        # Get digamma terms
        gamma_sum = gamma_params.sum(dim=1, keepdim=True)
        digamma_diff = torch.digamma(gamma_sum) - torch.digamma(gamma_params)
        
        # Score is digamma difference plus log of sample
        score = digamma_diff + torch.log(doc_topics + self.epsilon)
        
        return score


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


def train_bbvi(docs, n_topics, alpha0=0.1, eta0=0.01, n_iterations=600, 
               initial_lr=0.1, n_samples=50, device='cpu',
               scheduler_step_size=200, scheduler_gamma=0.5):
    """
    Train LDA model with vanilla Black-Box Variational Inference (BBVI)
    using PyTorch's autograd and AdaGrad with learning rate scheduling.
    """
    # Move data to specified device
    docs = docs.to(device)
    
    vocab_size = docs.shape[1]
    n_docs = docs.shape[0]
    
    # Initialize model and move to device
    model = LDABBVI(vocab_size, n_topics, alpha0, eta0).to(device)
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
    lr_values = []
    
    # Training loop
    for iteration in tqdm(range(n_iterations)):
        # Reset gradients
        topic_optimizer.zero_grad()
        doc_optimizer.zero_grad()
        
        # Compute gradient estimates using multiple MC samples
        topic_grads = torch.zeros_like(model.lambda_topics)
        doc_grads = torch.zeros_like(model.gamma)
        
        elbo_samples = []
        
        # Collect samples and compute score function gradients
        for s in range(n_samples):
            # Draw samples from variational distribution
            topics, doc_topics = model.sample_variational(1)
            topics = topics.squeeze(0)  # Remove batch dimension
            doc_topics = doc_topics.squeeze(0)
            
            # Compute log joint and log variational density
            log_joint = model.log_joint_prob(topics, doc_topics, docs)
            log_q = model.log_q(topics, doc_topics)
            
            # Store ELBO sample
            elbo_sample = log_joint - log_q
            elbo_samples.append(elbo_sample.item())
            
            # Compute score function for each parameter
            topic_score = model.topic_score(topics)
            doc_score = model.doc_score(doc_topics)
            
            # Multiply by ELBO estimate
            elbo_estimate = log_joint - log_q
            
            # Accumulate gradients (negative for minimization)
            topic_grads -= topic_score * elbo_estimate
            doc_grads -= doc_score * elbo_estimate
        
        # Average gradients over samples
        topic_grads /= n_samples
        doc_grads /= n_samples
        
        # Apply gradients manually
        model.lambda_topics.grad = topic_grads
        model.gamma.grad = doc_grads
        
        # Apply gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(model.lambda_topics, 5.0)
        # torch.nn.utils.clip_grad_norm_(model.gamma, 5.0)
        
        # Update parameters
        topic_optimizer.step()
        doc_optimizer.step()
        
        # Step the learning rate schedulers
        topic_scheduler.step()
        doc_scheduler.step()
        
        # Compute and store ELBO estimate
        current_elbo = sum(elbo_samples) / len(elbo_samples)
        elbo_values.append((iteration, current_elbo))
        
        # Track effective learning rates
        if hasattr(topic_optimizer, 'param_groups') and len(topic_optimizer.param_groups) > 0:
            param_state = list(topic_optimizer.state.values())[0] if topic_optimizer.state else {}
            if 'sum' in param_state:
                scheduler_factor = topic_scheduler.get_last_lr()[0] / initial_lr
                effective_lr = scheduler_factor * initial_lr / (torch.sqrt(param_state['sum'][0, 0]) + topic_optimizer.defaults['eps'])
                lr_values.append((iteration, effective_lr.item()))
        
        # Print progress
        if (iteration + 1) % 50 == 0:
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
    initial_lr = 0.11
    n_samples = 10  # Number of samples for score function estimator
    
    # Scheduler parameters
    scheduler_step_size = 200
    scheduler_gamma = 0.5

    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data
    print("Generating synthetic LDA data...")
    documents, true_topics, true_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, alpha0, eta0
    )
    docs = torch.tensor(documents, dtype=torch.float32)
    
    # Train model with vanilla BBVI using AdaGrad and learning rate scheduling
    print("Training LDA model with vanilla BBVI using AdaGrad and learning rate scheduling...")
    start = time.time()
    model, elbo_values, lr_values = train_bbvi(
        docs, n_topics, alpha0, eta0, n_iterations, initial_lr, n_samples, device,
        scheduler_step_size, scheduler_gamma
    )
    stop = time.time()
    
    # Extract iteration and ELBO values for plotting
    iterations, elbos = zip(*elbo_values)
    time_steps = np.linspace(0, stop-start, len(elbos))
    df = pd.DataFrame({"elbo": elbos, "time_steps": time_steps})
    df.to_csv("vanilla_bbvi_dataset.csv")
    
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
    plt.title('ELBO Convergence with BBVI using AdaGrad + Scheduler')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.grid(True)
    plt.legend()
    # plt.savefig('bbvi_elbo_trace.png')
    plt.show()
    
    # Plot learning rates
    if lr_values:
        lr_iterations, lr_rates = zip(*lr_values)
        plt.figure(figsize=(10, 4))
        plt.plot(lr_iterations, lr_rates)
        plt.title('AdaGrad Effective Learning Rate with Scheduler')
        plt.xlabel('Iteration')
        plt.ylabel('Effective Learning Rate')
        plt.grid(True)
        plt.yscale('log')  # Log scale to better visualize the decay
        # plt.savefig('bbvi_adagrad_lr_trace.png')
        plt.show()
    
    print("ELBO trace plot saved as 'bbvi_elbo_trace.png'")
    if lr_values:
        print("AdaGrad learning rate plot saved as 'bbvi_adagrad_lr_trace.png'")


if __name__ == "__main__":
    main()