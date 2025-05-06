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
        # for i in range(self.n_docs):
        #     for j in range(self.vocab_size):
        #         if docs[i, j] > 0:
        #             word_prob = 0
        #             for k in range(self.n_topics):
        #                 word_prob += doc_topics[i, k] * topics[k, j]
        #             log_lik += docs[i, j] * torch.log(word_prob + 1e-10)
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

def train_lda_bbvi(bow_matrix, n_topics, alpha0=1.0, eta0=1.0, n_epochs=100, batch_size=32, lr=0.01):
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
            print(f"Epoch {epoch+1}/{n_epochs}, ELBO: {avg_elbo:.4f}")
    plt.figure(figsize=(10, 5))
    plt.plot(elbo_history)
    plt.show()

    return model

if __name__ == "__main__":
    vocab_size = 1000
    n_topics = 10
    n_docs = 100
    doc_length = 70
    alpha0 = 0.1
    eta0 = 0.01
    documents, true_topics, tru_doc_topics = generate_lda_data(
        vocab_size, n_topics, n_docs, doc_length, alpha0, eta0
    )
    model = train_lda_bbvi(
        documents,
        n_topics=n_topics,
        alpha0=alpha0,
        eta0=eta0,
        n_epochs=600,
        batch_size=32,
        lr=0.01
    )