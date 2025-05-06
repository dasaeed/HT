import numpy as np
import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class FastBBVI:
    def __init__(self, n_topics, vocab_size, alpha0=0.1, eta0=0.01):
        """Initialize parameters for BBVI on LDA"""
        self.n_topics = n_topics
        self.vocab_size = vocab_size
        self.alpha0 = alpha0
        self.eta0 = eta0
        
        # Initialize global parameters (topics)
        self.topic_log_params = torch.nn.Parameter(
            torch.zeros((n_topics, vocab_size)) + np.log(eta0)
        )
        
        # Document parameters will be created when needed
        self.doc_log_params = None
        self.n_docs = None
    
    def setup_docs(self, n_docs):
        """Create document parameters"""
        self.n_docs = n_docs
        self.doc_log_params = torch.nn.Parameter(
            torch.zeros((n_docs, self.n_topics)) + np.log(self.alpha0)
        )
        return self.doc_log_params
    
    def sample_topics(self):
        """Sample from variational distribution of topics"""
        # Convert log parameters to natural parameters
        lambda_params = torch.exp(self.topic_log_params.detach())
        
        # Sample from Dirichlet
        samples = torch.zeros_like(lambda_params)
        for k in range(self.n_topics):
            # Sample from Gamma then normalize for Dirichlet
            gamma_samples = torch.distributions.Gamma(
                concentration=lambda_params[k], 
                rate=torch.ones_like(lambda_params[k])
            ).sample()
            samples[k] = gamma_samples / gamma_samples.sum()
        
        return samples
    
    def sample_doc_topics(self):
        """Sample from document-topic variational distribution"""
        # Convert log parameters to natural parameters
        gamma_params = torch.exp(self.doc_log_params.detach())
        
        # Sample from Dirichlet
        samples = torch.zeros_like(gamma_params)
        for i in range(self.n_docs):
            # Sample from Gamma then normalize for Dirichlet
            gamma_samples = torch.distributions.Gamma(
                concentration=gamma_params[i], 
                rate=torch.ones_like(gamma_params[i])
            ).sample()
            samples[i] = gamma_samples / gamma_samples.sum()
        
        return samples
    
    def dirichlet_log_prob(self, x, alpha):
        """Compute log probability of Dirichlet distribution"""
        log_gamma_sum = torch.lgamma(alpha.sum())
        log_gamma_terms = torch.lgamma(alpha).sum()
        log_prob_terms = ((alpha - 1) * torch.log(x.clamp(min=1e-10))).sum()
        return log_gamma_sum - log_gamma_terms + log_prob_terms
    
    def elbo_sample(self, docs):
        """Compute a single sample estimate of the ELBO"""
        # Sample from variational distributions
        topics = self.sample_topics()
        doc_topics = self.sample_doc_topics()
        
        # Get variational parameters
        lambda_params = torch.exp(self.topic_log_params)
        gamma_params = torch.exp(self.doc_log_params)
        
        # Define prior parameters
        topic_prior = torch.full((self.vocab_size,), self.eta0, device=docs.device)
        doc_prior = torch.full((self.n_topics,), self.alpha0, device=docs.device)
        
        # 1. Compute log prior for topics
        log_p_topics = 0.0
        for k in range(self.n_topics):
            log_p_topics += self.dirichlet_log_prob(topics[k], topic_prior)
        
        # 2. Compute log prior for document-topic proportions
        log_p_doc_topics = 0.0
        for i in range(self.n_docs):
            log_p_doc_topics += self.dirichlet_log_prob(doc_topics[i], doc_prior)
        
        # 3. Compute log likelihood for documents
        log_lik = 0.0
        for i in range(self.n_docs):
            word_probs = torch.zeros(self.vocab_size, device=docs.device)
            for k in range(self.n_topics):
                word_probs += doc_topics[i, k] * topics[k]
            
            word_probs = word_probs.clamp(min=1e-10)
            log_lik += (docs[i] * torch.log(word_probs)).sum()
        
        # 4. Compute log q(topics) (entropy term)
        log_q_topics = 0.0
        for k in range(self.n_topics):
            log_q_topics += self.dirichlet_log_prob(topics[k], lambda_params[k])
        
        # 5. Compute log q(doc_topics) (entropy term)
        log_q_doc_topics = 0.0
        for i in range(self.n_docs):
            log_q_doc_topics += self.dirichlet_log_prob(doc_topics[i], gamma_params[i])
        
        # Combine all terms for ELBO
        elbo = log_p_topics + log_p_doc_topics + log_lik - log_q_topics - log_q_doc_topics
        
        return elbo
    
    def compute_elbo(self, docs, n_samples=1):
        """Compute ELBO with multiple samples"""
        elbo_sum = 0.0
        for _ in range(n_samples):
            elbo_sum += self.elbo_sample(docs)
        return elbo_sum / n_samples
    
    def get_topic_distributions(self):
        """Get normalized topic distributions"""
        topic_params = torch.exp(self.topic_log_params.detach())
        topic_dist = topic_params / topic_params.sum(dim=1, keepdim=True)
        return topic_dist.numpy()


def generate_lda_data(vocab_size, n_topics, n_docs, doc_length, alpha0=1.0, eta0=1.0, seed=42):
    """Generate synthetic LDA data"""
    np.random.seed(seed)
    
    # Generate true topics
    true_topics = np.zeros((n_topics, vocab_size))
    for k in range(n_topics):
        true_topics[k] = np.random.dirichlet(np.ones(vocab_size) * eta0)
    
    # Generate documents
    documents = np.zeros((n_docs, vocab_size), dtype=np.float32)
    true_doc_topics = np.zeros((n_docs, n_topics))
    
    for d in range(n_docs):
        # Generate document-topic proportions
        theta = np.random.dirichlet(np.ones(n_topics) * alpha0)
        true_doc_topics[d] = theta
        
        # Generate document
        doc_len = np.random.poisson(doc_length)
        for _ in range(doc_len):
            # Sample topic
            z = np.random.choice(n_topics, p=theta)
            # Sample word
            w = np.random.choice(vocab_size, p=true_topics[z])
            # Add to document
            documents[d, w] += 1
    
    return documents, true_topics, true_doc_topics


def train_bbvi(docs, n_topics, alpha0=0.1, eta0=0.01, n_epochs=100, 
               doc_lr=0.1, topic_lr=0.01, n_samples=1, device='cpu',
               burnin_epochs=600, scheduler_patience=10):
    """
    Train LDA using BBVI with a burn-in period before applying learning rate scheduling
    
    Parameters:
    -----------
    docs : numpy.ndarray
        Document-word count matrix
    n_topics : int
        Number of topics to infer
    alpha0, eta0 : float
        Dirichlet prior parameters
    n_epochs : int
        Total number of epochs to train
    doc_lr, topic_lr : float
        Initial learning rates
    n_samples : int
        Number of Monte Carlo samples per gradient step
    device : str
        'cpu' or 'cuda'
    burnin_epochs : int
        Number of epochs to train before starting the learning rate scheduler
    scheduler_patience : int
        Number of epochs without improvement before reducing learning rate
    """
    # Create model
    model = FastBBVI(n_topics, docs.shape[1], alpha0, eta0)
    
    # Move data and model to device
    docs_tensor = torch.tensor(docs, dtype=torch.float32, device=device)
    model.topic_log_params = model.topic_log_params.to(device)
    model.setup_docs(docs.shape[0])
    model.doc_log_params = model.doc_log_params.to(device)
    
    # Create optimizers - separate for topics and documents
    topic_optimizer = Adam([model.topic_log_params], lr=topic_lr)
    doc_optimizer = Adam([model.doc_log_params], lr=doc_lr)
    
    # Add learning rate schedulers with more aggressive parameters
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    topic_scheduler = ReduceLROnPlateau(topic_optimizer, mode='max', factor=0.5, 
                                        patience=scheduler_patience, verbose=True,
                                        threshold=1e-4, threshold_mode='rel')
    doc_scheduler = ReduceLROnPlateau(doc_optimizer, mode='max', factor=0.5, 
                                      patience=scheduler_patience, verbose=True,
                                      threshold=1e-4, threshold_mode='rel')
    
    # Track ELBO
    elbo_history = []
    time_history = []
    start_time = time.time()
    best_elbo = float('-inf')
    scheduler_activated = False
    
    # Training loop
    print(f"Starting training with {burnin_epochs} burn-in epochs...")
    for epoch in tqdm(range(n_epochs)):
        # Step 1: Update document parameters
        doc_optimizer.zero_grad()
        elbo = model.compute_elbo(docs_tensor, n_samples)
        (-elbo).backward(retain_graph=True)  # Minimize negative ELBO
        doc_optimizer.step()
        
        # Step 2: Update topic parameters
        topic_optimizer.zero_grad()
        elbo = model.compute_elbo(docs_tensor, n_samples)
        (-elbo).backward()  # Minimize negative ELBO
        topic_optimizer.step()
        
        # Track ELBO and time
        elbo_val = elbo.item()
        elbo_history.append(elbo_val)
        time_history.append(time.time() - start_time)
        
        # Update best ELBO
        if elbo_val > best_elbo:
            best_elbo = elbo_val
        
        # Check if burn-in is complete and we need to activate schedulers
        if epoch == burnin_epochs and not scheduler_activated:
            # Reset best ELBO to current value to ensure schedulers work properly
            print(f"Burn-in complete at epoch {epoch+1}. Activating learning rate schedulers.")
            # Reset optimizer learning rates (optional)
            for param_group in topic_optimizer.param_groups:
                param_group['lr'] = topic_lr * 0.5  # Optional step down after burn-in
            for param_group in doc_optimizer.param_groups:
                param_group['lr'] = doc_lr * 0.5  # Optional step down after burn-in
            scheduler_activated = True
            
        # Update learning rate schedulers only after burn-in period
        if epoch >= burnin_epochs:
            topic_scheduler.step(elbo_val)
            doc_scheduler.step(elbo_val)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            time_elapsed = time_history[-1]
            print(f"Epoch {epoch+1}, ELBO: {elbo_val:.4f}, Best ELBO: {best_elbo:.4f}, Time: {time_elapsed:.1f}s")
            print(f"Topic LR: {topic_optimizer.param_groups[0]['lr']:.6f}, Doc LR: {doc_optimizer.param_groups[0]['lr']:.6f}")
            
            # Also print if we're still in burn-in
            if epoch < burnin_epochs:
                remaining = burnin_epochs - epoch - 1
                print(f"Still in burn-in period. {remaining} epochs remaining before scheduler activates.")
    
    return model, elbo_history, time_history


if __name__ == "__main__":
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate synthetic data (smaller dataset for faster testing)
    print("Generating synthetic data...")
    vocab_size = 500  # Reduced from 500
    n_topics = 5      # Reduced from 10
    n_docs = 50       # Reduced from 200
    doc_length = 50
    alpha0 = 0.1
    eta0 = 0.01
    
    documents, true_topics, true_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, alpha0, eta0
    )
    
    # Set training parameters
    n_epochs = 800     # Reduced from 100
    doc_lr = 0.1
    topic_lr = 0.01
    n_samples = 10
    
    # Train model
    print(f"Training LDA with {n_topics} topics on {n_docs} documents")
    model, elbo_history, time_history = train_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=n_epochs,
        doc_lr=doc_lr,
        topic_lr=topic_lr,
        n_samples=n_samples,
        device=device
    )
    
    # Plot ELBO trace
    plt.figure(figsize=(10, 6))
    plt.plot(elbo_history)
    plt.title("ELBO Trace")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.grid(True)
    plt.savefig("elbo_trace.png")
    plt.show()