import re
import numpy as np
import numpy.random as npr
from scipy.stats import mode
from scipy.special import gammaln, digamma
from scipy.stats import dirichlet as dir

def clean_text(text):
    with open("vocab.txt", "r") as file:
        vocab = [line.strip() for line in file.readlines()]
    vocab_set = set(vocab)
    contractions = {
        r"n\'t": 'nt',
        r"\'s": 's',     
        r"\'ve": 've',   
        r"\'re": 're',   
        r"\'m": 'm',     
        r"\'ll": 'll',   
        r"\'d": 'd'      
    }

    text = text.lower().strip()
    text = re.sub(r'[.,!?;:"``]', " ", text)
    words = []
    for word in text.split():
        if "'" in word:
            word_added = False
            for pattern, replacement in contractions.items():
                if word.endswith(pattern):
                    contracted = word.replace(pattern, replacement)
                    if contracted in vocab_set:
                        words.append(contracted)
                        word_added = True
                        break
            if not word_added:
                combined = word.replace("'", "")
                if combined in vocab_set:
                    words.append(combined)
        if "-" in word:
            combined = word.replace("-", "")
            if combined in vocab_set:
                words.append(combined)
                continue 
            parts = [p for p in word.split("-") if p in vocab_set]
            words.extend(parts)
        else:
            if word in vocab_set:
                words.append(word)
    return words


def init_var_params(X, K, V, rs_int=npr.randint(low=0, high=100)):
    rs = npr.RandomState(rs_int)
    N = len(X)
    lambd = rs.uniform(low=0.5, high=1.0, size=(K, V))
    gamma = np.ones((N, K))
    return np.log(lambd), np.log(gamma)

def sample_params(var_params):
    lambd, gamma = var_params
    K = lambd.shape[0]
    N = gamma.shape[0]

    beta = np.zeros_like(lambd)
    for k in range(K):
        beta[k] = dir.rvs(np.exp(lambd[k]))

    theta = np.zeros_like(gamma)
    for i in range(N):
        theta[i] = dir.rvs(np.exp(gamma[i]))
    return beta, theta

def log_dir(x, alpha):
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha)) + np.sum((alpha - 1) * np.log(x))

def log_var_approx(var_params, latent_params):
    lambd, gamma = var_params
    beta, theta = latent_params
    K = lambd.shape[0]
    N = gamma.shape[0]

    log_q_beta = sum(dir.logpdf(beta[k], np.exp(lambd[k])) for k in range(K))
    log_q_theta = sum(dir.logpdf(theta[i], np.exp(gamma[i])) for i in range(N))
    return log_q_beta + log_q_theta

def log_joint_prob(latent_params, X):
    beta, theta = latent_params
    K = beta.shape[0]
    N = theta.shape[0]
    Ms = [len(x_i) for x_i in X]

    log_p_beta = sum(dir.logpdf(beta[k], np.full(beta[k].shape, eta0)) for k in range(K))
    log_p_theta = sum(dir.logpdf(theta[i], np.full(theta[i].shape, alpha0)) for i in range(N))
    log_p_x = 0.0
    for i in range(N):
        for j in range(Ms[i]):
            x_ij = X[i][j]
            log_p_x += np.log(sum(theta[i, k] * beta[k, x_ij] for k in range(K)))
    return log_p_beta + log_p_theta + log_p_x

def score_dir(x, alpha):
    return digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)

def stoch_score_grad(var_params, latent_params):
    lambd, gamma = var_params
    K = lambd.shape[0]
    N = gamma.shape[0]
    beta, theta = latent_params
    grad_lambd, grad_gamma = np.zeros_like(lambd), np.zeros_like(gamma)
    for k in range(K):
        grad_lambd[k] = score_dir(beta[k], np.exp(lambd[k])) * np.exp(lambd[k])
    for i in range(N):
        grad_gamma[i] = score_dir(theta[i], np.exp(gamma[i])) * np.exp(gamma[i])

    return grad_lambd, grad_gamma

with open("vocab.txt", "r") as file:
    vocab = [line.strip() for line in file.readlines()]
vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}
V = len(vocab_to_idx)
X = []

with open("ap.txt", "r") as file:
    raw_text = file.read()
documents = re.findall(r"<TEXT>\n(.*?)\n </TEXT", raw_text, re.DOTALL)
for doc in documents:
    words = clean_text(doc)
    if len(words) > 100:
        X.append([vocab_to_idx[word] for word in words])
X = [np.asarray(x_i) for x_i in X]
N = len(X)
Ms = [len(x_i) for x_i in X]
K = 20

S = 5
eta0, alpha0 = 0.8, 0.5
eta, eps = 0.1, 1e-8
lambd, gamma = init_var_params(X, K, V)
G_lambda = np.zeros((V, V))
G_gamma = np.zeros((K, K))

for t in range(500):
    grad_lambda = np.zeros_like(lambd)
    grad_gamma = np.zeros_like(gamma)
    for s in range(S):
        beta_s, theta_s = sample_params((lambd, gamma))
        grad_lambda_s, grad_gamma_s = stoch_score_grad((lambd, gamma), (beta_s, theta_s))
        log_p, log_q = log_joint_prob((beta_s, theta_s), X), log_var_approx((lambd, gamma), (beta_s, theta_s))
        grad_lambda += grad_lambda_s * (log_p - log_q)
        grad_gamma += grad_gamma_s * (log_p - log_q)
    grad_lambda /= S
    grad_gamma /= S

    for k in range(K):
        G_lambda += np.outer(grad_lambda[k], grad_lambda[k])
    rho_lambda = eta * (G_lambda.diagonal() + eps)**(-0.5)
    lambd += rho_lambda * grad_lambda

    for i in range(N):
        G_gamma += np.outer(grad_gamma[i], grad_gamma[i])
    rho_gamma = eta * (G_gamma.diagonal() + eps)**(-0.5)
    gamma += rho_gamma * grad_gamma

    if t % 10 == 0:
        print(f"Iteration: {t}")


lambd_transf = np.exp(lambd)
word_topic_probs = lambd_transf / lambd_transf.sum(axis=1, keepdims=True)
top_words = {}
for k in range(word_topic_probs.shape[0]):
    top_idxs = np.argsort(word_topic_probs[k, :])[-10:][::-1]
    top_words[k] = [vocab[v] for v in top_idxs]

formatted_text = "Top 10 Words for Each Topic:\n\n"
for topic, words in top_words.items():
    formatted_text += f"Topic {topic + 1}: "
    formatted_text += ", ".join(words) + "\n\n"

print(formatted_text)