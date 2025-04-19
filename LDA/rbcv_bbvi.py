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
    
    def compute_log_p_for_dimension(self, dim_idx, param_type, topics, doc_topics, docs):
        topic_prior, doc_prior = self.get_prior_dirichlets()
        if param_type == "topics":
            topic_idx, word_idx = dim_idx // self.vocab_size, dim_idx % self.vocab_size
            log_p = topic_prior.log_prob(topics[topic_idx])
            for i in range(self.n_docs):
                if docs[i, word_idx] > 0:
                    word_prob = doc_topics[i, topic_idx] * topics[topic_idx, word_idx]
                    log_p += docs[i, word_idx] * torch.log(word_prob + 1e-10)
            return log_p
        else:
            doc_idx, topic_idx = dim_idx // self.n_topics, dim_idx % self.n_topics
            log_p = doc_prior.log_prob(doc_topics[doc_idx])
            word_probs = torch.matmul(doc_topics[doc_idx].unsqueeze(0), topics).squeeze(0)
            mask = docs[doc_idx] > 0
            if mask.sum() > 0:
                log_p += torch.sum(docs[doc_idx][mask] * torch.log(word_probs[mask] + 1e-10))
            return log_p
        
    def compute_log_joint_for_doc_markov_blanket(self, doc_idx, topics, doc_topics, docs):
        _, doc_prior = self.get_prior_dirichlets()
        log_p = doc_prior.log_prob(doc_topics[doc_idx])
        word_probs = torch.matmul(doc_topics[doc_idx].unsqueeze(0), topics).squeeze(0)
        mask = docs[doc_idx] > 0
        if mask.sum() > 0:
            log_p += torch.sum(docs[doc_idx][mask] * torch.log(word_probs[mask] + 1e-10))
        return log_p

    def compute_elbo(self, docs, n_samples=10):
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

    def forward(self, docs, n_samples=1):
        self.setup_doc_params(docs.shape[0])
        elbo_val = self.compute_elbo(docs, n_samples)
        return -elbo_val
    
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

def compute_score_function(dirichlet_dist, sample):
    alpha = dirichlet_dist.concentration
    digamma_sum = torch.digamma(torch.sum(alpha))
    digamma_alpha = torch.digamma(alpha)
    log_sample = torch.log(sample + 1e-10)
    score = digamma_sum - digamma_alpha + log_sample
    return score

def train_lda_rbcv_bbvi(bow_matrix, n_topics, alpha0=1.0, eta0=1.0, n_epochs=100, batch_size=32,
                        lr=0.01, n_samples=10, warmup_epochs=10):
    vocab_size = bow_matrix.shape[1]
    dataset = LDADataset(bow_matrix)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LDABBVI(vocab_size, n_topics, alpha0, eta0)
    topic_optimizer = Adam([model.topic_log_var], lr=lr)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(1, (n_epochs - warmup_epochs))
            return 0.5 * (1 + np.cos(np.pi * progress))
    topic_scheduler = torch.optim.lr_scheduler.LambdaLR(topic_optimizer, lr_lambda)
    elbo_history = []
    for epoch in tqdm(range(n_epochs)):
        epoch_elbo = 0
        for batch_idx, docs in enumerate(dataloader):
            model.setup_doc_params(docs.shape[0])
            doc_optimizer = Adam([model.doc_log_var], lr=topic_optimizer.param_groups[0]["lr"])
            all_params = []
            param_shapes = []
            param_sizes = []
            param_types = []

            param_shapes.append(model.topic_log_var.shape)
            param_sizes.append(model.topic_log_var.numel())
            param_types.extend(["topic"] * model.topic_log_var.numel())
            all_params.append(model.topic_log_var.view(-1))

            param_shapes.append(model.doc_log_var.shape)
            param_sizes.append(model.doc_log_var.numel())
            param_types.extend(["doc"] * model.doc_log_var.numel())
            all_params.append(model.doc_log_var.view(-1))

            all_params = torch.cat(all_params)
            total_dims = all_params.numel()
            n_samples_S = n_samples
            topic_q, doc_q = model.get_var_dirichlets()
            topics_samples = []
            doc_topics_samples = []
            for s in range(n_samples_S):
                topics = torch.stack([q.rsample() for q in topic_q])
                doc_topics = torch.stack([q.rsample() for q in doc_q])
                topics_samples.append(topics)
                doc_topics_samples.append(doc_topics)
            topic_gradient = torch.zeros_like(model.topic_log_var)
            doc_gradient = torch.zeros_like(model.doc_log_var)

            for d in range(total_dims):
                param_type = param_types[d]
                f_values = []
                h_values = []
                for s in range(n_samples_S):
                    topics = topics_samples[s]
                    doc_topics = doc_topics_samples[s]
                    if param_type == "topic":
                        topic_idx, word_idx = d // vocab_size, d % vocab_size
                        score = compute_score_function(topic_q[topic_idx], topics[topic_idx])[word_idx]
                        h_value = score
                    else:
                        doc_offset = model.topic_log_var.numel()
                        doc_d = d - doc_offset
                        doc_idx, topic_idx = doc_d // model.n_topics, doc_d % model.n_topics
                        score = compute_score_function(doc_q[doc_idx], doc_topics[doc_idx])[topic_idx]
                        h_value = score
                    log_p = model.compute_log_p_for_dimension(d, param_type, topics, doc_topics, docs)
                    if param_type == "topic":
                        topic_idx = d // vocab_size
                        log_q = topic_q[topic_idx].log_prob(topics[topic_idx])
                    else:
                        doc_offset = model.topic_log_var.numel()
                        doc_d = d - doc_offset
                        doc_idx = doc_d // model.n_topics
                        log_q = doc_q[doc_idx].log_prob(doc_topics[doc_idx])
                    f_value = h_value * (log_p - log_q)
                    f_values.append(f_value)
                    h_values.append(h_value)
                dim_gradient = 0
                for s in range(n_samples_S):
                    f_wo_s = f_values.copy()
                    h_wo_s = h_values.copy()
                    f_wo_s.pop(s)
                    h_wo_s.pop(s)
                    f_tensor = torch.stack(f_wo_s)
                    h_tensor = torch.stack(h_wo_s)

                    cov_fh = torch.mean(f_tensor * h_tensor) - torch.mean(f_tensor) * torch.mean(h_tensor)
                    var_h = torch.var(h_tensor) if h_tensor.numel() > 1 else torch.tensor(1.0)
                    a_s = cov_fh / (var_h + 1e-10)

                    topics = topics_samples[s]
                    doc_topics = doc_topics_samples[s]
                    if param_type == 'topic':
                        topic_idx, word_idx = d // vocab_size, d % vocab_size
                        score = compute_score_function(topic_q[topic_idx], topics[topic_idx])[word_idx]
                    else:
                        doc_offset = model.topic_log_var.numel()
                        doc_d = d - doc_offset
                        doc_idx, topic_idx = doc_d // model.n_topics, doc_d % model.n_topics
                        score = compute_score_function(doc_q[doc_idx], doc_topics[doc_idx])[topic_idx]
                    log_p = model.compute_log_p_for_dimension(d, param_type, topics, doc_topics, docs)
                    if param_type == 'topic':
                        topic_idx = d // vocab_size
                        log_q = topic_q[topic_idx].log_prob(topics[topic_idx])
                    else:
                        doc_offset = model.topic_log_var.numel()
                        doc_d = d - doc_offset
                        doc_idx = doc_d // model.n_topics
                        log_q = doc_q[doc_idx].log_prob(doc_topics[doc_idx])
                    grad_sample = score * (log_p - log_q) - a_s * score
                    dim_gradient += grad_sample

                    avg_gradient = dim_gradient / n_samples_S
                    if param_type == 'topic':
                        topic_idx, word_idx = d // vocab_size, d % vocab_size
                        topic_gradient[topic_idx, word_idx] = avg_gradient
                    else:
                        doc_offset = model.topic_log_var.numel()
                        doc_d = d - doc_offset
                        doc_idx, topic_idx = doc_d // model.n_topics, doc_d % model.n_topics
                        doc_gradient[doc_idx, topic_idx] = avg_gradient
                
                topic_optimizer.zero_grad()
                model.topic_log_var.grad = -topic_gradient
                topic_optimizer.step()

                for _ in range(3):
                    doc_optimizer.zero_grad()
                    model.doc_log_var.grad = -doc_gradient
                    doc_optimizer.step()
                    
                    topic_q, doc_q = model.get_var_dirichlets()
                    doc_gradient = torch.zeros_like(model.doc_log_var)
                    topics = torch.stack([q.rsample() for q in topic_q])
                    doc_topics = torch.stack([q.rsample() for q in doc_q])
                    
                    for i in range(model.n_docs):
                        scores = compute_score_function(doc_q[i], doc_topics[i])
                        log_p = model.compute_log_joint_for_doc_markov_blanket(i, topics, doc_topics, docs)
                        log_q = doc_q[i].log_prob(doc_topics[i])
                        doc_gradient[i] = scores * (log_p - log_q)
                
            with torch.no_grad():
                current_elbo = model.compute_elbo(docs)
                epoch_elbo += current_elbo.item()
        topic_scheduler.step()
        avg_elbo = epoch_elbo / len(dataloader)
        elbo_history.append(avg_elbo)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}, LR: {topic_optimizer.param_groups[0]['lr']:.6f}")
    plt.figure(figsize=(10, 5))
    plt.plot(elbo_history)
    plt.xlabel('Epoch')
    plt.ylabel('ELBO')
    plt.title('ELBO History during Training')
    plt.show()
    return model

if __name__ == "__main__":
    vocab_size = 100
    n_topics = 5
    n_docs = 200
    doc_length = 50
    alpha0 = 0.1 
    eta0 = 0.01   
    
    documents, true_topics, true_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, alpha0, eta0
    )

    model = train_lda_rbcv_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=100,
        batch_size=32,
        lr=0.01,
        n_samples=10,
        warmup_epochs=10
    )