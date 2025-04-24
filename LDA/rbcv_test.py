import numpy as np
import torch
import torch.distributions as dist
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "STIXGeneral"

class LDADataset(Dataset):
    def __init__(self, bow_matrix):
        self.bow_matrix = bow_matrix
    
    def __len__(self):
        return self.bow_matrix.shape[0]
    
    def __getitem__(self, idx):
        return self.bow_matrix[idx]
    
class LDABBVI(torch.nn.Module):
    def __init__(self, vocab_size, n_topics, alpha0=1.0, eta0=1.0):
        super(LDABBVI, self).__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.alpha0 = alpha0
        self.eta0 = eta0

        # Log-parameterization for unconstrained optimization
        self.topic_log_var = torch.nn.Parameter(
            torch.randn(n_topics, vocab_size) * 0.01 + np.log(1.0 / vocab_size)
        )
        self.doc_log_var = None
        self.n_docs = None

    def setup_doc_params(self, n_docs):
        self.n_docs = n_docs
        self.doc_log_var = torch.nn.Parameter(
            torch.zeros(n_docs, self.n_topics) + np.log(1.0 / self.n_topics)
        )

    def get_topic_dist(self):
        return torch.softmax(self.topic_log_var, dim=1)
    
    def get_doc_topic_prop(self):
        return torch.softmax(self.doc_log_var, dim=1)
    
    def get_var_params(self):
        lambda_params = torch.exp(self.topic_log_var)
        gamma_params = torch.exp(self.doc_log_var)
        return lambda_params, gamma_params
    
    def get_prior_dirichlets(self):
        topic_prior = dist.Dirichlet(
            torch.ones(self.vocab_size) * self.eta0
        )
        doc_prior = dist.Dirichlet(
            torch.ones(self.n_topics) * self.alpha0
        )
        return topic_prior, doc_prior
    
    def get_var_dirichlets(self):
        lambda_params, gamma_params = self.get_var_params()
        topic_q = []
        for k in range(self.n_topics):
            topic_q.append(dist.Dirichlet(lambda_params[k]))
        doc_q = []
        for i in range(self.n_docs):
            doc_q.append(dist.Dirichlet(gamma_params[i]))
        return topic_q, doc_q
    
    def log_joint_prob(self, topics, doc_topics, docs):
        topic_prior, doc_prior = self.get_prior_dirichlets()
        log_p_topics = 0
        for k in range(self.n_topics):
            log_p_topics += topic_prior.log_prob(topics[k])
        log_p_doc_topics = 0
        for i in range(self.n_docs):
            log_p_doc_topics += doc_prior.log_prob(doc_topics[i])
        log_lik = 0
        for i in range(self.n_docs):
            word_probs = torch.matmul(doc_topics[i].unsqueeze(0), topics).squeeze(0)
            mask = docs[i] > 0
            if mask.sum() > 0:
                log_lik += torch.sum(docs[i][mask] * torch.log(word_probs[mask] + 1e-10))
        return log_p_topics + log_p_doc_topics + log_lik
    
    def compute_score_function(self, latent_vars):
        """
        Compute the score function (gradient of log q w.r.t. variational parameters)
        for both topic and document variational parameters.
        """
        topics, doc_topics = latent_vars
        lambda_params, gamma_params = self.get_var_params()
        
        # Score function for topic parameters
        topic_score = []
        for k in range(self.n_topics):
            lambda_k = lambda_params[k]
            lambda_sum = lambda_k.sum()
            digamma_sum = torch.digamma(lambda_sum)
            digamma_lambda = torch.digamma(lambda_k)
            log_topics = torch.log(topics[k] + 1e-10)
            score_k = lambda_k * (digamma_sum - digamma_lambda + log_topics)
            topic_score.append(score_k)
        
        # Score function for document parameters
        doc_score = []
        for i in range(self.n_docs):
            gamma_i = gamma_params[i]
            gamma_sum = gamma_i.sum()
            digamma_sum = torch.digamma(gamma_sum)
            digamma_gamma = torch.digamma(gamma_i)
            log_doc_topics = torch.log(doc_topics[i] + 1e-10)
            score_i = gamma_i * (digamma_sum - digamma_gamma + log_doc_topics)
            doc_score.append(score_i)
            
        return topic_score, doc_score
    
    def elbo(self, docs, n_samples=10):
        topic_q, doc_q = self.get_var_dirichlets()
        elbo_val = 0
        for _ in range(n_samples):
            topics = torch.stack([topic_q[k].rsample() for k in range(self.n_topics)])
            doc_topics = torch.stack([doc_q[i].rsample() for i in range(self.n_docs)])
            log_joint = self.log_joint_prob(topics, doc_topics, docs)
            log_q = 0
            for k in range(self.n_topics):
                log_q += topic_q[k].log_prob(topics[k])
            for i in range(self.n_docs):
                log_q += doc_q[i].log_prob(doc_topics[i])
            elbo_val += log_joint - log_q
        return elbo_val / n_samples
    
    def forward(self, docs, n_samples=10):
        self.setup_doc_params(docs.shape[0])
        return -self.elbo(docs, n_samples)

class LDARBCVBBVI(LDABBVI):
    """
    LDA with Rao-Blackwellized Control Variate Black-Box Variational Inference
    """
    def __init__(self, vocab_size, n_topics, alpha0=1.0, eta0=1.0):
        super(LDARBCVBBVI, self).__init__(vocab_size, n_topics, alpha0, eta0)
    
    def compute_elbo_gradient_estimate(self, docs, n_samples=10):
        """
        Compute gradient estimate of ELBO using Rao-Blackwellization and control variates
        """
        topic_q, doc_q = self.get_var_dirichlets()
        lambda_params, gamma_params = self.get_var_params()
        
        # Initialize gradient accumulators
        topic_grad = torch.zeros_like(self.topic_log_var)
        doc_grad = torch.zeros_like(self.doc_log_var)
        
        # Data structures for samples and control variate computation
        topic_samples = []  # Store samples to compute leave-one-out estimates
        doc_samples = []
        score_topic_samples = []  # Store score function values
        score_doc_samples = []
        elbo_samples = []  # Store ELBO values
        
        # Generate samples and compute ELBO values
        for s in range(n_samples):
            # Sample from variational distribution
            topics = torch.stack([topic_q[k].rsample() for k in range(self.n_topics)])
            doc_topics = torch.stack([doc_q[i].rsample() for i in range(self.n_docs)])
            
            topic_samples.append(topics)
            doc_samples.append(doc_topics)
            
            # Compute log joint probability (log p)
            log_joint = self.log_joint_prob(topics, doc_topics, docs)
            
            # Compute log variational distribution (log q)
            log_q = 0
            for k in range(self.n_topics):
                log_q += topic_q[k].log_prob(topics[k])
            for i in range(self.n_docs):
                log_q += doc_q[i].log_prob(doc_topics[i])
            
            # Instantaneous ELBO = log p - log q
            elbo_sample = log_joint - log_q
            elbo_samples.append(elbo_sample)
            
            # Compute score function (gradient of log q wrt variational parameters)
            topic_scores = []
            for k in range(self.n_topics):
                lambda_k = lambda_params[k]
                lambda_sum = lambda_k.sum()
                digamma_sum = torch.digamma(lambda_sum)
                digamma_lambda = torch.digamma(lambda_k)
                log_topics_k = torch.log(topics[k] + 1e-10)
                score_k = lambda_k * (digamma_sum - digamma_lambda + log_topics_k)
                topic_scores.append(score_k)
            
            doc_scores = []
            for i in range(self.n_docs):
                gamma_i = gamma_params[i]
                gamma_sum = gamma_i.sum()
                digamma_sum = torch.digamma(gamma_sum)
                digamma_gamma = torch.digamma(gamma_i)
                log_doc_topics_i = torch.log(doc_topics[i] + 1e-10)
                score_i = gamma_i * (digamma_sum - digamma_gamma + log_doc_topics_i)
                doc_scores.append(score_i)
            
            score_topic_samples.append(topic_scores)
            score_doc_samples.append(doc_scores)
        
        # Compute leave-one-out control variate estimates for topic parameters
        for k in range(self.n_topics):
            for s in range(n_samples):
                # Extract function values f = score * ELBO for all samples except s
                f_values = []
                h_values = []
                
                for j in range(n_samples):
                    if j != s:
                        f_values.append(score_topic_samples[j][k] * elbo_samples[j])
                        h_values.append(score_topic_samples[j][k])
                
                # Compute optimal control variate coefficient
                if len(f_values) > 0:
                    f_tensor = torch.stack(f_values)
                    h_tensor = torch.stack(h_values)
                    
                    f_mean = f_tensor.mean(dim=0)
                    h_mean = h_tensor.mean(dim=0)
                    
                    # Covariance between f and h
                    cov_f_h = ((f_tensor - f_mean) * (h_tensor - h_mean)).mean(dim=0)
                    
                    # Variance of h
                    var_h = ((h_tensor - h_mean) ** 2).mean(dim=0)
                    
                    # Compute a* (with numerical stability)
                    a_star = torch.where(var_h > 1e-8, cov_f_h / var_h, torch.zeros_like(var_h))
                    
                    # Apply control variate
                    f_s = score_topic_samples[s][k] * elbo_samples[s]
                    h_s = score_topic_samples[s][k]
                    
                    # Control variate reduced gradient: f - a*(h - E[h]) = f - a*h + a*E[h]
                    # Since E[h] = 0 for score function, this simplifies to f - a*h
                    grad_s = f_s - a_star * h_s
                    
                    # Accumulate gradient
                    topic_grad[k] += grad_s / n_samples
        
        # Compute leave-one-out control variate estimates for document parameters
        for i in range(self.n_docs):
            for s in range(n_samples):
                # Extract function values f = score * ELBO for all samples except s
                f_values = []
                h_values = []
                
                for j in range(n_samples):
                    if j != s:
                        f_values.append(score_doc_samples[j][i] * elbo_samples[j])
                        h_values.append(score_doc_samples[j][i])
                
                # Compute optimal control variate coefficient
                if len(f_values) > 0:
                    f_tensor = torch.stack(f_values)
                    h_tensor = torch.stack(h_values)
                    
                    f_mean = f_tensor.mean(dim=0)
                    h_mean = h_tensor.mean(dim=0)
                    
                    # Covariance between f and h
                    cov_f_h = ((f_tensor - f_mean) * (h_tensor - h_mean)).mean(dim=0)
                    
                    # Variance of h
                    var_h = ((h_tensor - h_mean) ** 2).mean(dim=0)
                    
                    # Compute a* (with numerical stability)
                    a_star = torch.where(var_h > 1e-8, cov_f_h / var_h, torch.zeros_like(var_h))
                    
                    # Apply control variate
                    f_s = score_doc_samples[s][i] * elbo_samples[s]
                    h_s = score_doc_samples[s][i]
                    
                    # Control variate reduced gradient
                    grad_s = f_s - a_star * h_s
                    
                    # Accumulate gradient
                    doc_grad[i] += grad_s / n_samples
        
        return topic_grad, doc_grad
    
    def compute_cv_coefficient(self, f_samples, h_samples):
        """
        Compute optimal control variate coefficient:
        a* = Cov(f, h) / Var(h)
        """
        # Skip computation if no samples
        if len(f_samples) == 0 or len(h_samples) == 0:
            return 0.0
            
        # Stack tensors for vectorized operations
        try:
            f = torch.stack(f_samples)
            h = torch.stack(h_samples)
        except:
            # Handle case where tensors have different shapes
            return torch.zeros_like(f_samples[0]) if f_samples else 0.0
        
        # Compute means
        f_mean = f.mean(dim=0, keepdim=True)
        h_mean = h.mean(dim=0, keepdim=True)
        
        # Compute covariance: Cov(f, h) = E[(f - E[f])(h - E[h])]
        cov = ((f - f_mean) * (h - h_mean)).mean(dim=0)
        
        # Compute variance of h: Var(h) = E[(h - E[h])²]
        var_h = ((h - h_mean) ** 2).mean(dim=0)
        
        # Compute optimal coefficient with numerical stability
        # If variance is too small, set coefficient to zero to avoid instability
        a = torch.zeros_like(var_h)
        mask = var_h > 1e-8
        a[mask] = cov[mask] / var_h[mask]
        
        return a

def generate_lda_data(vocab_size, n_topics, n_docs, doc_length, alpha0=1.0, eta0=1.0):
    true_topics = np.zeros((n_topics, vocab_size))
    for k in range(n_topics):
        true_topics[k] = np.random.dirichlet(np.ones(vocab_size) * eta0)
    documents = np.zeros((n_docs, vocab_size), dtype=int)
    true_doc_topics = np.zeros((n_docs, n_topics))
    for d in range(n_docs):
        theta = np.random.dirichlet(np.ones(n_topics) * alpha0)
        true_doc_topics[d] = theta
        doc_len = np.random.poisson(doc_length)
        for _ in range(doc_len):
            z = np.random.choice(n_topics, p=theta)
            w = np.random.choice(vocab_size, p=true_topics[z])
            documents[d, w] += 1
    return documents, true_topics, true_doc_topics

def train_lda_rbcv_bbvi(bow_matrix, n_topics, alpha0=1.0, eta0=1.0, n_epochs=100, batch_size=32, lr=0.01, n_samples=10):
    vocab_size = bow_matrix.shape[1]
    dataset = LDADataset(bow_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model with RBCV BBVI
    model = LDARBCVBBVI(vocab_size, n_topics, alpha0, eta0)
    
    # Optimizer for topic parameters (document parameters will be optimized per batch)
    topic_optimizer = Adam([{"params": model.topic_log_var}], lr=lr)
    scheduler = StepLR(topic_optimizer, step_size=10, gamma=0.9)
    
    elbo_history = []
    
    for epoch in tqdm(range(n_epochs)):
        epoch_elbo = 0
        
        for batch_idx, docs in enumerate(dataloader):
            # Set up document parameters for this batch
            model.setup_doc_params(docs.shape[0])
            
            # Optimizer for document parameters
            doc_optimizer = Adam([model.doc_log_var], lr=lr)
            
            # Optimize document parameters with multiple steps
            for _ in range(5):
                # Important: Zero out gradients before computation
                doc_optimizer.zero_grad()
                
                # Compute RBCV gradient estimates for document parameters
                _, doc_grad = model.compute_elbo_gradient_estimate(docs, n_samples)
                
                # Set gradients manually - use positive gradients for maximizing
                # We want to maximize ELBO, so gradients should be positive
                model.doc_log_var.grad = doc_grad
                
                # Update parameters
                doc_optimizer.step()
            
            # Optimize topic parameters
            topic_optimizer.zero_grad()
            
            # Compute RBCV gradient estimates for topic parameters
            topic_grad, _ = model.compute_elbo_gradient_estimate(docs, n_samples)
            
            # Set gradients manually - use positive gradients for maximizing
            model.topic_log_var.grad = topic_grad
            
            # Update parameters
            topic_optimizer.step()
            
            # Compute ELBO for monitoring (ELBO is the negative of the loss from forward)
            with torch.no_grad():
                elbo_val = -model(docs, n_samples=5).item()  # Use fewer samples for evaluation
                epoch_elbo += elbo_val
        
        # Update learning rate
        scheduler.step()
        
        # Average ELBO over batches
        avg_elbo = epoch_elbo / len(dataloader)
        elbo_history.append(avg_elbo)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}")
    
    # Plot ELBO history
    plt.figure(figsize=(10, 5))
    plt.plot(elbo_history)
    plt.title("ELBO Convergence with RBCV BBVI")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.grid(True)
    plt.show()
    
    return model, elbo_history

def evaluate_topics(model, true_topics, vocabulary=None):
    """
    Evaluate the learned topics against true topics using topic coherence
    and visualize the top words for each topic
    """
    learned_topics = model.get_topic_dist().detach().numpy()
    n_topics = learned_topics.shape[0]
    
    if vocabulary is None:
        vocabulary = [f"word_{i}" for i in range(learned_topics.shape[1])]
    
    # Visualize top words for each topic
    top_k = 10
    fig, axs = plt.subplots(n_topics, 2, figsize=(15, 3*n_topics))
    
    for k in range(n_topics):
        # True topic
        true_top_idx = np.argsort(-true_topics[k])[:top_k]
        true_top_words = [vocabulary[idx] for idx in true_top_idx]
        true_weights = true_topics[k][true_top_idx]
        
        # Learned topic
        learned_top_idx = np.argsort(-learned_topics[k])[:top_k]
        learned_top_words = [vocabulary[idx] for idx in learned_top_idx]
        learned_weights = learned_topics[k][learned_top_idx]
        
        # Plot true topic
        axs[k, 0].barh(range(top_k), true_weights, align='center')
        axs[k, 0].set_yticks(range(top_k))
        axs[k, 0].set_yticklabels(true_top_words)
        axs[k, 0].set_title(f"True Topic {k+1}")
        
        # Plot learned topic
        axs[k, 1].barh(range(top_k), learned_weights, align='center')
        axs[k, 1].set_yticks(range(top_k))
        axs[k, 1].set_yticklabels(learned_top_words)
        axs[k, 1].set_title(f"Learned Topic {k+1}")
    
    plt.tight_layout()
    plt.show()

def compare_vanilla_and_rbcv_bbvi(documents, n_topics, alpha0, eta0, n_epochs=200, batch_size=32, lr=0.01):
    """
    Run both vanilla BBVI and RBCV BBVI with same data and parameters, and compare results
    """
    from bbvi_pytorch import train_lda_bbvi, LDABBVI
    
    # For fair comparison, modify train_lda_bbvi to return ELBO history
    def train_lda_bbvi_with_history(bow_matrix, n_topics, alpha0=1.0, eta0=1.0, n_epochs=100, batch_size=32, lr=0.01):
        vocab_size = bow_matrix.shape[1]
        dataset = LDADataset(bow_matrix)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = LDABBVI(vocab_size, n_topics, alpha0, eta0)
        optimizer = Adam([{"params": model.topic_log_var}], lr=lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        elbo_history = []
        
        for epoch in tqdm(range(n_epochs)):
            epoch_elbo = 0
            for batch_idx, docs in enumerate(dataloader):
                optimizer.zero_grad()
                model.setup_doc_params(docs.shape[0])
                doc_optimizer = Adam([model.doc_log_var], lr=lr)
                for _ in range(5):
                    doc_optimizer.zero_grad()
                    loss = model(docs)
                    loss.backward(retain_graph=True)
                    doc_optimizer.step()
                loss = model(docs)
                loss.backward()
                optimizer.step()
                epoch_elbo -= loss.item()
                scheduler.step()
            
            avg_elbo = epoch_elbo / len(dataloader)
            elbo_history.append(avg_elbo)

            if (epoch +1) % 10 == 0:
                print(f"Vanilla BBVI - Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}")
        
        return model, elbo_history
    
    # Convert to tensor if not already
    if not isinstance(documents, torch.Tensor):
        documents = torch.tensor(documents, dtype=torch.float32)
    
    # Train with vanilla BBVI
    print("Training LDA model with vanilla BBVI...")
    vanilla_model, vanilla_elbo_history = train_lda_bbvi_with_history(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr
    )
    
    # Train with RBCV BBVI
    print("\nTraining LDA model with RBCV BBVI...")
    rbcv_model, rbcv_elbo_history = train_lda_rbcv_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_samples=10
    )
    
    # Plot comparison of ELBO convergence
    plt.figure(figsize=(12, 6))
    plt.plot(vanilla_elbo_history, label='Vanilla BBVI')
    plt.plot(rbcv_elbo_history, label='RBCV BBVI')
    plt.title("ELBO Convergence Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return vanilla_model, rbcv_model, vanilla_elbo_history, rbcv_elbo_history

if __name__ == "__main__":
    # Parameters for synthetic data generation
    vocab_size = 500
    n_topics = 10
    n_docs = 200
    doc_length = 70
    alpha0 = 0.1
    eta0 = 0.01
    
    # Generate synthetic data
    print("Generating synthetic LDA data...")
    documents, true_topics, true_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, alpha0, eta0
    )
    
    # Compare vanilla BBVI and RBCV BBVI
    print("Running comparison between vanilla BBVI and RBCV BBVI...")
    vanilla_model, rbcv_model, vanilla_elbo, rbcv_elbo = compare_vanilla_and_rbcv_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=100,  # Reduced for demonstration
        batch_size=32,
        lr=0.01
    )
