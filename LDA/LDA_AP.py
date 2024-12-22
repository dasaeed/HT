import os
import time
import pandas as pd
import numpy as np
import numpy.random as npr
import copy
import re
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from scipy.special import digamma, loggamma

def log_sum_exp(vec):
    vec_max = np.max(vec, axis=0)
    exp_vec = np.exp(vec - vec_max)
    sum_exp_vec = np.sum(exp_vec)
    log_sum_exp = np.log(sum_exp_vec) + vec_max
    return log_sum_exp

def load_data():
    with open("vocab.txt", "r") as f:
        raw_lines = f.readlines()

    idx_to_words = [word.strip() for word in raw_lines]
    V = len(idx_to_words)

    with open("ap_bow.txt", "r") as f:
        raw_lines = f.readlines()
        N = len(raw_lines)
        
    articles = np.zeros((N, V))
    nonzero_idxs = []

    for i in tqdm(range(N)):
        split = raw_lines[i].split(" ")
        n_words = int(split[0])
        split = split[1:]

        article = np.zeros((V,))
        nonzero_idx = []

        for bow in split:
            bow = bow.strip()
            word_idx, count = bow.split(":")
            nonzero_idx.append(int(word_idx))
            article[int(word_idx)] = count

        try:
            assert(len(nonzero_idx) == n_words)
        except:
            raise AssertionError(f"{len(nonzero_idx)}, {n_words}")

        articles[i] = article
        nonzero_idxs.append(sorted(nonzero_idx))
    
    return idx_to_words, articles, nonzero_idxs

def init_variational_params(articles, K):
    N, V = articles.shape
    LAMBDA = np.random.uniform(low=0.01, high=1.0, size=(K, V))
    GAMMA = np.ones((N, K))
    PHI = []
    for article in articles:
        n_words = np.sum((article > 0).astype("int32"))
        article_PHI = np.ones((n_words, K))
        article_PHI = article_PHI / K

        PHI.append(article_PHI)
    return LAMBDA, GAMMA, PHI

def compute_ELBO(LAMBDA, GAMMA, PHI, articles, nonzero_idxs, K):
    ELBO = 0
    N, V = articles.shape

    E_log_p_beta = 0
    for k in range(K):
        E_log_p_beta += (ETA-1) * np.sum(digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k])))
    ELBO += E_log_p_beta

    E_log_p_theta = 0
    for i in range(N):
        E_log_p_theta += (ALPHA-1) * np.sum(digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i])))
    ELBO += E_log_p_theta

    E_log_p_xz = 0
    for i in range(N):
        article = articles[i]
        nonzero_idx = nonzero_idxs[i]
        corr_idx = 0
        for idx in nonzero_idx:
            E_log_p_xz += article[idx] * np.sum(PHI[i][corr_idx] * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))
            E_log_p_xz += article[idx] * np.sum(PHI[i][corr_idx] * (digamma(LAMBDA[:,idx]) - digamma(np.sum(LAMBDA, axis=1))))
            corr_idx += 1

        assert(corr_idx == len(nonzero_idx))
    ELBO += E_log_p_xz

    E_log_q_beta = 0
    for k in range(K):
        E_log_q_beta += -loggamma(np.sum(LAMBDA[k])) + np.sum(loggamma(LAMBDA[k]))
        E_log_q_beta += -np.sum((LAMBDA[k]-1) * (digamma(LAMBDA[k]) - digamma(np.sum(LAMBDA[k]))))
    ELBO += E_log_q_beta

    E_log_q_theta = 0
    for i in range(N):
        E_log_q_theta += -loggamma(np.sum(GAMMA[i])) + np.sum(loggamma(GAMMA[i]))
        E_log_q_theta += -np.sum((GAMMA[i]-1) * (digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))))
    ELBO += E_log_q_theta

    E_log_q_z = 0
    for i in range(N):
        article = articles[i]
        nonzero_idx = nonzero_idxs[i]
        corr_idx = 0
        for idx in nonzero_idx:
            E_log_q_z += -article[idx] * np.sum(PHI[i][corr_idx] * np.log(PHI[i][corr_idx]))
            corr_idx += 1

        assert(corr_idx == len(nonzero_idx))
    ELBO += E_log_q_z

    return ELBO


start = time.time()
idx_to_words, articles, nonzero_idxs = load_data()
N, V = articles.shape
K = 20
ETA = 100 / V
ALPHA = 1 / K
LAMBDA, GAMMA, PHI = init_variational_params(articles, K)

ELBOs = []
prev_ELBO = -np.inf
curr_ELBO = compute_ELBO(LAMBDA, GAMMA, PHI, articles, nonzero_idxs, K)
ELBOs.append(curr_ELBO)
print(f"Initial ELBO: {ELBOs[0]}")

max_iterations = 100
tol = 10e-3
LAMBDA_t = copy.deepcopy(LAMBDA)
GAMMA_t = copy.deepcopy(GAMMA)
PHI_t = copy.deepcopy(PHI)

for t in range(max_iterations):
    print(f"Iteration {t}")
    for i in tqdm(range(N), desc="Updating PHI and GAMMA"):
        article = articles[i]
        nonzero_idx = nonzero_idxs[i]
        GAMMA_i_t = copy.deepcopy(GAMMA_t[i])
        corr_idx = 0
        for idx in nonzero_idx:
            log_PHI_ij = np.zeros((K,))
            for k in range(K):
                LAMBDA_k_t = copy.deepcopy(LAMBDA_t[k])
                exp_propto = digamma(GAMMA_i_t[k]) - digamma(np.sum(GAMMA_i_t))
                exp_propto += digamma(LAMBDA_k_t[idx]) - digamma(np.sum(LAMBDA_k_t))
                log_PHI_ij[k] = exp_propto
            PHI_ij = np.exp(log_PHI_ij - log_sum_exp(log_PHI_ij))
            PHI_t[i][corr_idx] = PHI_ij
            corr_idx += 1
        GAMMA_i_t = np.zeros((K,)) + ALPHA
        for k in range(K):
            GAMMA_i_t[k] += np.sum(article[nonzero_idx] * PHI_t[i][:,k])
        GAMMA_t[i] = GAMMA_i_t

    for k in tqdm(range(K), desc="Updating LAMBDA"):
        LAMBDA_k_t = np.zeros((V,)) + ETA
        for i in range(N):
            article = articles[i]
            nonzero_idx = nonzero_idxs[i]
            corr_idx = 0
            for idx in nonzero_idx:
                LAMBDA_k_t[idx] += article[idx] * PHI_t[i][corr_idx][k]
                corr_idx += 1
            LAMBDA_t[k] = LAMBDA_k_t

    prev_ELBO = curr_ELBO
    curr_ELBO = compute_ELBO(LAMBDA_t, GAMMA_t, PHI_t, articles, nonzero_idxs, K)
    print(f"Current ELBO: {curr_ELBO} | Change in ELBO: {curr_ELBO - prev_ELBO}\n")

    if abs(curr_ELBO - prev_ELBO) < tol:
        break
stop = time.time()
