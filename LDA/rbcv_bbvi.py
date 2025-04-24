import numpy as np
import torch
import torch.distributions as dist
import torch.nn.functional as F
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
        # Apply softplus to ensure all values are positive
        lambda_params = F.softplus(self.topic_log_var) + 1e-6
        gamma_params = F.softplus(self.doc_log_var) + 1e-6
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
    
    def compute_score_function(self, latent_vars, robust=True):
        """
        Compute the score function (gradient of log q w.r.t. variational parameters)
        for both topic and document variational parameters.
        
        Args:
            latent_vars: Tuple of (topics, doc_topics)
            robust: Whether to use robust computation with gradient clipping
            
        Returns:
            Tuple of (topic_score, doc_score)
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
            
            if robust:
                # Clip to avoid numerical instability
                score_k = torch.clamp(score_k, -10.0, 10.0)
                
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
            
            if robust:
                # Clip to avoid numerical instability
                score_i = torch.clamp(score_i, -10.0, 10.0)
                
            doc_score.append(score_i)
            
        return topic_score, doc_score
    
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
            try:
                topics = torch.stack([topic_q[k].rsample() for k in range(self.n_topics)])
                doc_topics = torch.stack([doc_q[i].rsample() for i in range(self.n_docs)])
            except ValueError as e:
                # Handle potential sampling error
                print(f"Sampling error: {e}")
                # Return zero gradients in case of error
                return torch.zeros_like(self.topic_log_var), torch.zeros_like(self.doc_log_var)
            
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
            topic_scores, doc_scores = self.compute_score_function((topics, doc_topics), robust=True)
            
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
                    try:
                        f_tensor = torch.stack(f_values)
                        h_tensor = torch.stack(h_values)
                        
                        f_mean = f_tensor.mean(dim=0)
                        h_mean = h_tensor.mean(dim=0)
                        
                        # Covariance between f and h
                        cov_f_h = ((f_tensor - f_mean) * (h_tensor - h_mean)).mean(dim=0)
                        
                        # Variance of h
                        var_h = ((h_tensor - h_mean) ** 2).mean(dim=0)
                        
                        # Compute a* (with numerical stability)
                        a_star = torch.zeros_like(var_h)
                        mask = var_h > 1e-8
                        a_star[mask] = cov_f_h[mask] / var_h[mask]
                        
                        # Ensure coefficient isn't too large
                        a_star = torch.clamp(a_star, -5.0, 5.0)
                        
                        # Apply control variate
                        f_s = score_topic_samples[s][k] * elbo_samples[s]
                        h_s = score_topic_samples[s][k]
                        
                        # Control variate reduced gradient
                        grad_s = f_s - a_star * h_s
                        
                        # Accumulate gradient (with safety clipping)
                        grad_s = torch.clamp(grad_s, -10.0, 10.0)
                        topic_grad[k] += grad_s / n_samples
                    except Exception as e:
                        print(f"Error in control variate computation for topic {k}: {e}")
                        # Skip this sample
                        continue
        
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
                    try:
                        f_tensor = torch.stack(f_values)
                        h_tensor = torch.stack(h_values)
                        
                        f_mean = f_tensor.mean(dim=0)
                        h_mean = h_tensor.mean(dim=0)
                        
                        # Covariance between f and h
                        cov_f_h = ((f_tensor - f_mean) * (h_tensor - h_mean)).mean(dim=0)
                        
                        # Variance of h
                        var_h = ((h_tensor - h_mean) ** 2).mean(dim=0)
                        
                        # Compute a* (with numerical stability)
                        a_star = torch.zeros_like(var_h)
                        mask = var_h > 1e-8
                        a_star[mask] = cov_f_h[mask] / var_h[mask]
                        
                        # Ensure coefficient isn't too large
                        a_star = torch.clamp(a_star, -5.0, 5.0)
                        
                        # Apply control variate
                        f_s = score_doc_samples[s][i] * elbo_samples[s]
                        h_s = score_doc_samples[s][i]
                        
                        # Control variate reduced gradient
                        grad_s = f_s - a_star * h_s
                        
                        # Accumulate gradient (with safety clipping)
                        grad_s = torch.clamp(grad_s, -10.0, 10.0)
                        doc_grad[i] += grad_s / n_samples
                    except Exception as e:
                        print(f"Error in control variate computation for document {i}: {e}")
                        # Skip this sample
                        continue
        
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
        
        # Ensure coefficient isn't too large
        a = torch.clamp(a, -5.0, 5.0)
        
        return a
        
    def analyze_control_variate_effectiveness(self, docs, n_samples=100):
        """
        Analyze how well the control variates are working by computing variance reduction metrics
        """
        topic_q, doc_q = self.get_var_dirichlets()
        lambda_params, gamma_params = self.get_var_params()
        
        # Collect raw gradient samples and control variate reduced samples
        raw_gradients = []
        cv_reduced_gradients = []
        control_variate_coefficients = []
        
        for s in range(n_samples):
            # Sample from variational distribution
            try:
                topics = torch.stack([topic_q[k].rsample() for k in range(self.n_topics)])
                doc_topics = torch.stack([doc_q[i].rsample() for i in range(self.n_docs)])
            except ValueError as e:
                print(f"Sampling error in analysis: {e}")
                continue
            
            # Compute log joint and log q
            log_joint = self.log_joint_prob(topics, doc_topics, docs)
            log_q = sum(topic_q[k].log_prob(topics[k]) for k in range(self.n_topics)) + \
                   sum(doc_q[i].log_prob(doc_topics[i]) for i in range(self.n_docs))
            
            # ELBO for this sample
            elbo_sample = log_joint - log_q
            
            # Compute score function for one topic parameter (for analysis)
            k = 0  # Analyze first topic
            lambda_k = lambda_params[k]
            lambda_sum = lambda_k.sum()
            digamma_sum = torch.digamma(lambda_sum)
            digamma_lambda = torch.digamma(lambda_k)
            log_topics_k = torch.log(topics[k] + 1e-10)
            score_k = lambda_k * (digamma_sum - digamma_lambda + log_topics_k)
            
            # Clip to avoid numerical issues
            score_k = torch.clamp(score_k, -10.0, 10.0)
            
            # Raw gradient (without control variate)
            raw_grad = score_k * elbo_sample
            raw_gradients.append(raw_grad.detach())
            
            # Store for control variate computation
            if s % 10 == 0 and len(raw_gradients) > 1:  # Compute CV coefficient less frequently
                try:
                    cv_coeff = self.compute_cv_coefficient(
                        raw_gradients[:-1], 
                        [g / elbo_sample if torch.abs(elbo_sample) > 1e-10 else g for g in raw_gradients[:-1]]
                    )
                    control_variate_coefficients.append(cv_coeff.detach())
                    
                    # Apply control variate
                    cv_grad = raw_grad - cv_coeff * score_k
                    cv_reduced_gradients.append(cv_grad.detach())
                except Exception as e:
                    print(f"Error in CV analysis: {e}")
                    continue
        
        # Analyze variance reduction
        if len(raw_gradients) == 0:
            print("No valid samples for variance analysis")
            return {'raw_variance': None, 'cv_variance': None, 'variance_reduction': None}
            
        try:
            raw_var = torch.stack(raw_gradients).var(dim=0)
            cv_var = torch.stack(cv_reduced_gradients).var(dim=0) if cv_reduced_gradients else None
        
            # Plot analysis
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot raw gradient samples
            raw_grads_np = torch.stack(raw_gradients).numpy()
            ax1.plot(raw_grads_np[:, :10])  # Show first 10 dimensions
            ax1.set_title("Raw Gradient Samples (First 10 Dims)")
            ax1.set_xlabel("Sample")
            ax1.set_ylabel("Gradient Value")
            
            # Plot CV-reduced gradient samples
            if cv_reduced_gradients:
                cv_grads_np = torch.stack(cv_reduced_gradients).numpy()
                ax2.plot(cv_grads_np[:, :10])  # Show first 10 dimensions
                ax2.set_title("CV-Reduced Gradient Samples (First 10 Dims)")
                ax2.set_xlabel("Sample")
                ax2.set_ylabel("Gradient Value")
            
            # Plot variance reduction
            if cv_var is not None:
                reduction = (raw_var - cv_var) / raw_var
                ax3.bar(range(10), reduction[:10].numpy())
                ax3.set_title("Variance Reduction (First 10 Dims)")
                ax3.set_xlabel("Dimension")
                ax3.set_ylabel("Relative Variance Reduction")
                ax3.axhline(y=0, color='r', linestyle='--')
            
            # Plot control variate coefficients
            if control_variate_coefficients:
                cv_coeffs_np = torch.stack(control_variate_coefficients).numpy()
                ax4.plot(cv_coeffs_np[:, :10])  # Show first 10 dimensions
                ax4.set_title("Control Variate Coefficients (First 10 Dims)")
                ax4.set_xlabel("Iteration")
                ax4.set_ylabel("CV Coefficient")
            
            plt.tight_layout()
            plt.show()
            
            # Return metrics
            return {
                'raw_variance': raw_var.mean().item(),
                'cv_variance': cv_var.mean().item() if cv_var is not None else None,
                'variance_reduction': ((raw_var - cv_var) / raw_var).mean().item() if cv_var is not None else None
            }
        except Exception as e:
            print(f"Error in variance analysis: {e}")
            return {'raw_variance': None, 'cv_variance': None, 'variance_reduction': None}

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

def train_lda_rbcv_bbvi(bow_matrix, n_topics, alpha0=1.0, eta0=1.0, n_epochs=100, batch_size=32, lr=0.01, n_samples=10, use_adam_state=True):
    vocab_size = bow_matrix.shape[1]
    dataset = LDADataset(bow_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model with RBCV BBVI
    model = LDARBCVBBVI(vocab_size, n_topics, alpha0, eta0)
    
    # Create optimizers
    if use_adam_state:
        # Regular Adam optimizer with smaller learning rate for stability
        topic_optimizer = Adam([{"params": model.topic_log_var}], lr=lr)
    else:
        # SGD for manual gradient updates (use smaller learning rate for SGD)
        topic_optimizer = torch.optim.SGD([{"params": model.topic_log_var}], lr=lr * 0.1)
    
    scheduler = StepLR(topic_optimizer, step_size=10, gamma=0.9)
    
    elbo_history = []
    variance_history = []  # Track variance of gradient estimates
    
    for epoch in tqdm(range(n_epochs)):
        epoch_elbo = 0
        epoch_variance = 0
        
        for batch_idx, docs in enumerate(dataloader):
            # Set up document parameters for this batch
            model.setup_doc_params(docs.shape[0])
            
            # Optimizer for document parameters
            if use_adam_state:
                doc_optimizer = Adam([model.doc_log_var], lr=lr)
            else:
                # SGD with smaller learning rate
                doc_optimizer = torch.optim.SGD([model.doc_log_var], lr=lr * 0.1)
            
            # Optimize document parameters with multiple steps
            for _ in range(5):
                # Important: Zero out gradients before computation
                doc_optimizer.zero_grad()
                
                try:
                    # Compute RBCV gradient estimates for document parameters
                    _, doc_grad = model.compute_elbo_gradient_estimate(docs, n_samples)
                    
                    if use_adam_state:
                        # Set gradients for Adam to use its internal state
                        model.doc_log_var.grad = doc_grad
                        doc_optimizer.step()
                    else:
                        # Direct parameter update for SGD with gradient clipping
                        with torch.no_grad():
                            # Clip gradients to prevent extreme values
                            doc_grad_clipped = torch.clamp(doc_grad, -1.0, 1.0)
                            model.doc_log_var.data += lr * 0.1 * doc_grad_clipped
                except Exception as e:
                    print(f"Error in document parameter optimization (batch {batch_idx}): {e}")
                    continue  # Skip this step if there's an error
            
            # Optimize topic parameters
            topic_optimizer.zero_grad()
            
            try:
                # Compute RBCV gradient estimates for topic parameters
                topic_grad, _ = model.compute_elbo_gradient_estimate(docs, n_samples)
                
                # Track gradient variance
                grad_norm = torch.norm(topic_grad).item()
                epoch_variance += grad_norm
                
                if use_adam_state:
                    # Set gradients for Adam to use its internal state
                    model.topic_log_var.grad = topic_grad
                    topic_optimizer.step()
                else:
                    # Direct parameter update for SGD with gradient clipping
                    with torch.no_grad():
                        # Clip gradients to prevent extreme values
                        topic_grad_clipped = torch.clamp(topic_grad, -1.0, 1.0)
                        model.topic_log_var.data += lr * 0.1 * topic_grad_clipped
            except Exception as e:
                print(f"Error in topic parameter optimization (batch {batch_idx}): {e}")
                continue  # Skip this step if there's an error
            
            # Compute ELBO for monitoring (ELBO is the negative of the loss from forward)
            with torch.no_grad():
                try:
                    elbo_val = -model(docs, n_samples=5).item()  # Use fewer samples for evaluation
                    epoch_elbo += elbo_val
                except Exception as e:
                    print(f"Error computing ELBO (batch {batch_idx}): {e}")
                    continue
        
        # Update learning rate
        scheduler.step()
        
        # Average ELBO over batches
        avg_elbo = epoch_elbo / max(len(dataloader), 1)  # Avoid division by zero
        avg_variance = epoch_variance / max(len(dataloader), 1)  # Avoid division by zero
        elbo_history.append(avg_elbo)
        variance_history.append(avg_variance)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}, Grad Variance: {avg_variance:.4f}")
    
    # Plot ELBO history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(elbo_history)
    ax1.set_title("ELBO Convergence with RBCV BBVI")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO")
    ax1.grid(True)
    
    ax2.plot(variance_history)
    ax2.set_title("Gradient Variance")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Norm")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return model, elbo_history, variance_history

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
        
        avg_elbo = epoch_elbo / max(len(dataloader), 1)  # Avoid division by zero
        elbo_history.append(avg_elbo)

        if (epoch +1) % 10 == 0:
            print(f"Vanilla BBVI - Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}")
    
    return model, elbo_history

def compare_vanilla_and_rbcv_bbvi(documents, n_topics, alpha0, eta0, n_epochs=200, batch_size=32, lr=0.01):
    """
    Run both vanilla BBVI and RBCV BBVI with same data and parameters, and compare results
    """
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
    
    # Train with RBCV BBVI using Adam
    print("\nTraining LDA model with RBCV BBVI (Adam)...")
    rbcv_model_adam, rbcv_elbo_history_adam, rbcv_variance_adam = train_lda_rbcv_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_samples=10,
        use_adam_state=True
    )
    
    # Train with RBCV BBVI using SGD
    print("\nTraining LDA model with RBCV BBVI (SGD)...")
    rbcv_model_sgd, rbcv_elbo_history_sgd, rbcv_variance_sgd = train_lda_rbcv_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        n_samples=10,
        use_adam_state=False
    )
    
    # Plot comparison of ELBO convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # ELBO comparison
    ax1.plot(vanilla_elbo_history, label='Vanilla BBVI')
    ax1.plot(rbcv_elbo_history_adam, label='RBCV BBVI (Adam)')
    ax1.plot(rbcv_elbo_history_sgd, label='RBCV BBVI (SGD)')
    ax1.set_title("ELBO Convergence Comparison")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO")
    ax1.legend()
    ax1.grid(True)
    
    # Variance comparison
    ax2.plot(rbcv_variance_adam, label='RBCV BBVI (Adam)')
    ax2.plot(rbcv_variance_sgd, label='RBCV BBVI (SGD)')
    ax2.set_title("Gradient Variance Comparison")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Norm")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Plot relative improvement
    plt.figure(figsize=(10, 6))
    epochs = np.arange(len(vanilla_elbo_history))
    relative_improvement_adam = (np.array(rbcv_elbo_history_adam) - np.array(vanilla_elbo_history)) / np.abs(np.array(vanilla_elbo_history))
    relative_improvement_sgd = (np.array(rbcv_elbo_history_sgd) - np.array(vanilla_elbo_history)) / np.abs(np.array(vanilla_elbo_history))
    
    plt.plot(epochs, relative_improvement_adam * 100, label='RBCV BBVI (Adam) vs Vanilla')
    plt.plot(epochs, relative_improvement_sgd * 100, label='RBCV BBVI (SGD) vs Vanilla')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.title("Relative Improvement of RBCV BBVI over Vanilla BBVI")
    plt.xlabel("Epoch")
    plt.ylabel("Relative Improvement (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return vanilla_model, rbcv_model_adam, rbcv_model_sgd, vanilla_elbo_history, rbcv_elbo_history_adam, rbcv_elbo_history_sgd

def step_size_analysis(documents, n_topics, alpha0, eta0, learning_rates=[0.001, 0.01, 0.05, 0.1], n_epochs=50, batch_size=32):
    """
    Analyze the effect of different step sizes on RBCV BBVI performance
    """
    if not isinstance(documents, torch.Tensor):
        documents = torch.tensor(documents, dtype=torch.float32)
    
    # Compare with both Adam and SGD
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Results storage
    adam_results = {}
    sgd_results = {}
    
    # Test each learning rate
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        
        # Train with RBCV BBVI using Adam
        print(f"  Training with Adam...")
        _, rbcv_elbo_adam, _ = train_lda_rbcv_bbvi(
            documents,
            n_topics=n_topics,
            alpha0=alpha0,
            eta0=eta0,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_samples=10,
            use_adam_state=True
        )
        adam_results[lr] = rbcv_elbo_adam
        
        # Train with RBCV BBVI using SGD
        print(f"  Training with SGD...")
        _, rbcv_elbo_sgd, _ = train_lda_rbcv_bbvi(
            documents,
            n_topics=n_topics,
            alpha0=alpha0,
            eta0=eta0,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_samples=10,
            use_adam_state=False
        )
        sgd_results[lr] = rbcv_elbo_sgd
    
    # Plot results for Adam
    for lr, elbo in adam_results.items():
        axs[0].plot(elbo, label=f'lr={lr}')
    axs[0].set_title("RBCV BBVI with Adam Optimizer")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("ELBO")
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot results for SGD
    for lr, elbo in sgd_results.items():
        axs[1].plot(elbo, label=f'lr={lr}')
    axs[1].set_title("RBCV BBVI with SGD Optimizer")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("ELBO")
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return best results
    best_lr_adam = max(adam_results.items(), key=lambda x: x[1][-1])[0]
    best_lr_sgd = max(sgd_results.items(), key=lambda x: x[1][-1])[0]
    
    print(f"\nBest learning rate for Adam: {best_lr_adam} (Final ELBO: {adam_results[best_lr_adam][-1]:.4f})")
    print(f"Best learning rate for SGD: {best_lr_sgd} (Final ELBO: {sgd_results[best_lr_sgd][-1]:.4f})")
    
    return adam_results, sgd_results

def samples_analysis(documents, n_topics, alpha0, eta0, n_samples_list=[5, 10, 20, 50], n_epochs=50, batch_size=32, lr=0.01):
    """
    Analyze the effect of different numbers of Monte Carlo samples on RBCV BBVI performance
    """
    if not isinstance(documents, torch.Tensor):
        documents = torch.tensor(documents, dtype=torch.float32)
    
    # Results storage
    results = {}
    variance_results = {}
    
    # Test each number of samples
    for n_samples in n_samples_list:
        print(f"\nTesting with {n_samples} Monte Carlo samples")
        
        # Train with RBCV BBVI
        _, elbo_history, variance_history = train_lda_rbcv_bbvi(
            documents,
            n_topics=n_topics,
            alpha0=alpha0,
            eta0=eta0,
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            n_samples=n_samples,
            use_adam_state=True
        )
        results[n_samples] = elbo_history
        variance_results[n_samples] = variance_history
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot ELBO
    for n_samples, elbo in results.items():
        ax1.plot(elbo, label=f'samples={n_samples}')
    ax1.set_title("Effect of Number of Monte Carlo Samples on ELBO")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("ELBO")
    ax1.legend()
    ax1.grid(True)
    
    # Plot gradient variance
    for n_samples, variance in variance_results.items():
        ax2.plot(variance, label=f'samples={n_samples}')
    ax2.set_title("Effect of Number of Monte Carlo Samples on Gradient Variance")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Gradient Norm")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Return best results
    best_n_samples = max(results.items(), key=lambda x: x[1][-1])[0]
    print(f"\nBest number of samples: {best_n_samples} (Final ELBO: {results[best_n_samples][-1]:.4f})")
    
    return results, variance_results

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
    documents_tensor = torch.tensor(documents, dtype=torch.float32)
    
    # Analysis choice menu
    print("\nChoose analysis to run:")
    print("1. Compare vanilla BBVI and RBCV BBVI (Adam and SGD)")
    print("2. Analyze control variate effectiveness")
    print("3. Analyze effect of learning rate")
    print("4. Analyze effect of number of Monte Carlo samples")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == '1':
        # Compare vanilla BBVI and RBCV BBVI with different optimizers
        print("Running comparison between vanilla BBVI and RBCV BBVI...")
        vanilla_model, rbcv_model_adam, rbcv_model_sgd, vanilla_elbo, rbcv_elbo_adam, rbcv_elbo_sgd = compare_vanilla_and_rbcv_bbvi(
            documents,
            n_topics=n_topics,
            alpha0=alpha0,
            eta0=eta0,
            n_epochs=100,
            batch_size=32,
            lr=0.01
        )
        
        # Evaluate topics
        print("\nComparing learned topics:")
        print("Vanilla BBVI:")
        evaluate_topics(vanilla_model, true_topics)
        print("\nRBCV BBVI (Adam):")
        evaluate_topics(rbcv_model_adam, true_topics)
        print("\nRBCV BBVI (SGD):")
        evaluate_topics(rbcv_model_sgd, true_topics)
        
    elif choice == '2':
        # Analyze control variate effectiveness
        print("Analyzing control variate effectiveness...")
        model = LDARBCVBBVI(vocab_size, n_topics, alpha0, eta0)
        model.setup_doc_params(documents_tensor.shape[0])
        
        # Initialize parameters
        with torch.no_grad():
            # Initialize with non-random values for consistency
            model.topic_log_var.data = torch.zeros_like(model.topic_log_var) + np.log(1.0 / vocab_size)
            model.doc_log_var.data = torch.zeros_like(model.doc_log_var) + np.log(1.0 / n_topics)
        
        # Analyze
        metrics = model.analyze_control_variate_effectiveness(documents_tensor, n_samples=100)
        print("\nVariance Reduction Metrics:")
        print(f"Raw gradient variance: {metrics['raw_variance']:.6f}")
        print(f"CV-reduced gradient variance: {metrics['cv_variance']:.6f}")
        print(f"Relative variance reduction: {metrics['variance_reduction']*100:.2f}%")
        
    elif choice == '3':
        # Analyze effect of learning rate
        print("Analyzing effect of learning rate...")
        adam_results, sgd_results = step_size_analysis(
            documents_tensor,
            n_topics=n_topics,
            alpha0=alpha0,
            eta0=eta0,
            learning_rates=[0.001, 0.005, 0.01, 0.05, 0.1],
            n_epochs=50,
            batch_size=32
        )
        
    elif choice == '4':
        # Analyze effect of number of Monte Carlo samples
        print("Analyzing effect of number of Monte Carlo samples...")
        elbo_results, variance_results = samples_analysis(
            documents_tensor,
            n_topics=n_topics,
            alpha0=alpha0,
            eta0=eta0,
            n_samples_list=[5, 10, 20, 50],
            n_epochs=50,
            batch_size=32,
            lr=0.01
        )
        
    else:
        print("Invalid choice. Exiting.")