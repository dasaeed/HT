import re
import numpy as np
import numpy.random as npr
from scipy.stats import mode
from scipy.special import gammaln, digamma, logsumexp
from scipy.stats import dirichlet as dir
from scipy.stats import multinomial as multinom
from tqdm import tqdm
import matplotlib.pyplot as plt

rs = npr.RandomState(0)
K, V, N = 10, 300, 30
eta0, alpha0 = 1 / K, 1 / K
Ms = rs.poisson(60, size=N)
S = 500
lr = 1e-3
eps = 1e-6

def generate_lda(K, V, N, Ms, eta0=(1 / K), alpha0=(1 / K), rs_int=npr.randint(low=0, high=100)):
    eta0_vec = np.ones(V) * eta0
    alpha0_vec = np.ones(K) * alpha0

    beta = dir.rvs(eta0_vec, size=K)
    theta = dir.rvs(alpha0_vec, size=N)
    X = []
    for i in range(N):
        x_i = np.zeros(Ms[i], dtype=int)
        for j in range(Ms[i]):
            z_ij = np.argmax(multinom.rvs(1, p=theta[i]))
            x_ij = np.argmax(multinom.rvs(1, p=beta[z_ij]))
            x_i[j] = x_ij
        X.append(x_i)
    return X

def init_var_params(X, K, V, rs_int=npr.randint(low=0, high=100)):
    rs = npr.RandomState(rs_int)
    N = len(X)
    log_lambd = np.log(rs.uniform(low=0.1, high=1.0, size=(K, V)))
    log_gamma = np.log(rs.uniform(low=0.1, high=1.0, size=(N, K)))
    return log_lambd, log_gamma

def sample_params(var_params):
    log_lambd, log_gamma = var_params
    beta = np.zeros_like(log_lambd)
    theta = np.zeros_like(log_gamma)
    for k in range(K):
        beta[k] = dir.rvs(np.exp(log_lambd[k]))[0]
    for i in range(N):
        theta[i] = dir.rvs(np.exp(log_gamma[i]))[0]

    # log_gamm_beta = np.log(npr.gamma(np.exp(log_lambd), 1))
    # log_probs_beta = log_gamm_beta - logsumexp(log_gamm_beta, axis=1)[:, None]
    # beta = np.exp(log_probs_beta)

    # log_gamm_gamma = np.log(npr.gamma(np.exp(log_gamma), 1))
    # log_probs_gamma = log_gamm_gamma - logsumexp(log_gamm_gamma, axis=1)[:, None]
    # theta = np.exp(log_probs_gamma)
    return beta, theta

def log_var_approx(var_params, latent_params):
    log_lambd, log_gamma = var_params
    beta, theta = latent_params
    K = log_lambd.shape[0]
    N = log_gamma.shape[0]

    log_q_beta = sum(dir.logpdf(beta[k], np.exp(log_lambd[k])) for k in range(K))
    log_q_theta = sum(dir.logpdf(theta[i], np.exp(log_gamma[i])) for i in range(N))
    return log_q_beta + log_q_theta

def log_joint_prob(latent_params, X):
    beta, theta = latent_params
    K, V = beta.shape
    N = theta.shape[0]
    eta0_vec = np.ones(V) * eta0
    alpha0_vec = np.ones(K) * alpha0

    log_p_beta = sum(dir.logpdf(beta[k], eta0_vec) for k in range(K))
    log_p_theta = sum(dir.logpdf(theta[i], alpha0_vec) for i in range(N))
    log_p_x = 0.0
    for _, (theta_i, x_i) in enumerate(zip(theta, X)):
        beta_xi = beta[:, x_i]
        log_word_probs = np.log(np.sum(theta_i[:, None] * beta_xi, axis=0))
        log_p_x += np.sum(log_word_probs)
    return log_p_beta + log_p_theta + log_p_x

def score_dir(x, alpha):
    return digamma(np.sum(alpha)) - digamma(alpha) + np.log(x)

def score_var_dist(var_params, latent_params):
    log_lambd, log_gamma = var_params
    K = log_lambd.shape[0]
    N = log_gamma.shape[0]
    beta, theta = latent_params
    grad_lambd, grad_gamma = np.zeros_like(log_lambd), np.zeros_like(log_gamma)
    for k in range(K):
        grad_lambd[k] = score_dir(beta[k], np.exp(log_lambd[k]))
    for i in range(N):
        grad_gamma[i] = score_dir(theta[i], np.exp(log_gamma[i]))
    return grad_lambd, grad_gamma

def estimate_ELBO(var_params, X, S):
    log_lambd, log_gamma = var_params
    ELBO = 0.0
    for _ in range(S):
        beta_s, theta_s = sample_params((log_lambd, log_gamma))
        log_p = log_joint_prob((beta_s, theta_s), X)
        log_q = log_var_approx((log_lambd, log_gamma), (beta_s, theta_s))
        ELBO += (log_p - log_q)
    return ELBO / S

X = generate_lda(K, V, N, Ms, eta0=eta0, alpha0=alpha0, rs_int=0)
lambd, gamma = init_var_params(X, K, V, 0)
G_lambd, G_gamma = np.zeros_like(lambd), np.zeros_like(gamma)
max_iters = 10001
elbos = []

for t in range(max_iters):
    betas = []
    thetas = []
    for _ in range(S):
        beta_s, theta_s = sample_params((lambd, gamma))
        betas.append(beta_s)
        thetas.append(theta_s)

    stoch_score_grad_lambd = np.zeros_like(lambd)
    stoch_score_grad_gamma = np.zeros_like(gamma)
    for s in tqdm(range(S), desc=f"Iteration {t} | Calculating stochastic score gradient"):
        score_lambd, score_gamma = score_var_dist((lambd, gamma), (betas[s], thetas[s]))
        log_p = log_joint_prob((betas[s], thetas[s]), X)
        log_q = log_var_approx((lambd, gamma), (betas[s], thetas[s]))
        stoch_score_grad_lambd += score_lambd * (log_p - log_q)
        stoch_score_grad_gamma += score_gamma * (log_p - log_q)
    stoch_score_grad_lambd /= S
    stoch_score_grad_gamma /= S
    G_lambd += np.square(stoch_score_grad_lambd)
    lambd = lambd + (lr * stoch_score_grad_lambd / (G_lambd + eps))
    G_gamma += np.square(stoch_score_grad_gamma)
    gamma = gamma + (lr * stoch_score_grad_gamma / (G_gamma + eps))
    
    if t % 100 == 0:
        mc_elbo = estimate_ELBO((lambd, gamma), X, S)
        elbos.append(mc_elbo)
        print(f"\nMC ELBO: {estimate_ELBO((lambd, gamma), X, S)}\n")

fig, axs = plt.subplots()
axs.plot(np.arange(len(elbos)), elbos)
fig.show()
plt.savefig("hold.png")