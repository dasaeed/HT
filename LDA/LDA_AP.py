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

def load_documents():
    with open("vocab.txt", "r") as file:
        raw_lines = file.readlines()
    idx_to_words = [word.strip() for word in raw_lines]
    V = len(idx_to_words)

    with open("AP_BoW.txt", "r") as file:
        raw_lines = file.readlines()
        N = len(raw_lines)
    documents = np.zeros((N, V))
    nonzero_idxs = []

    for i in tqdm(range(N)):
        BoW_doc = raw_lines[i].split(" ")
        M = int(BoW_doc[0])
        BoW_doc = BoW_doc[1:]
        document = np.zeros((V,))
        nonzero_idx = []
        for BoW in BoW_doc:
            BoW = BoW.strip()
            word_idx, count = BoW.split(":")
            nonzero_idx.append(int(word_idx))
            document[int(word_idx)] = count
        assert(len(nonzero_idx) == M)

        documents[i] = document
        nonzero_idxs.append(sorted(nonzero_idx))
    
    return idx_to_words, documents, nonzero_idxs

def init_variational_params(documents, K):
    N, V = documents.shape
    LAMBDA = np.random.uniform(low=0.01, high=1.0, size=(K, V))
    GAMMA = np.ones((N, K))
    PHI = []
    for document in documents:
        M = np.sum((document > 0).astype("int32"))
        document_PHI = np.ones((M, K))
        document_PHI = document_PHI / K
        PHI.append(document_PHI)
        
    return LAMBDA, GAMMA, PHI

def compute_ELBO(LAMBDA, GAMMA, PHI, documents, nonzero_idxs, K):
    ELBO = 0
    N, _ = documents.shape

    E_log_p_BETA = np.sum((ETA-1) * (digamma(LAMBDA) - digamma(np.sum(LAMBDA, axis=1, keepdims=True))))
    ELBO += E_log_p_BETA

    E_log_p_THETA = np.sum((ALPHA-1) * (digamma(GAMMA) - digamma(np.sum(GAMMA, axis=1, keepdims=True))))
    ELBO += E_log_p_THETA

    E_log_p_x_z = 0
    for i in range(N):
        document = documents[i]
        nonzero_idx = nonzero_idxs[i]
        word_idx = 0
        for idx in nonzero_idx:
            E_log_p_x_z += np.sum(PHI[i][word_idx] * (digamma(GAMMA[i])-digamma(np.sum(GAMMA[i])))) \
                + np.sum(PHI[i][word_idx] * (digamma(LAMBDA[:, idx])-digamma(np.sum(LAMBDA, axis=1))))
            word_idx += 1
    ELBO += E_log_p_x_z

    E_log_q_BETA = np.sum(-loggamma(np.sum(LAMBDA, axis=1)) + np.sum(loggamma(LAMBDA), axis=1) \
        - np.sum((LAMBDA - 1) * (digamma(LAMBDA) - digamma(np.sum(LAMBDA, axis=1, keepdims=True))), axis=1))
    ELBO += E_log_q_BETA

    E_log_q_THETA = np.sum(-loggamma(np.sum(GAMMA, axis=1)) + np.sum(loggamma(GAMMA), axis=1) \
        - np.sum((GAMMA - 1) * (digamma(GAMMA) - digamma(np.sum(GAMMA, axis=1, keepdims=True))), axis=1))
    ELBO += E_log_q_THETA

    E_log_q_z = 0
    for i in range(N):
        document = documents[i]
        nonzero_idx = nonzero_idxs[i]
        word_idx = 0
        for idx in nonzero_idx:
            E_log_q_z += np.sum(PHI[i][word_idx] * np.log(PHI[i][word_idx]))
            word_idx += 1
    ELBO += E_log_q_z

    return ELBO

start = time.time()
idx_to_words, documents, nonzero_idxs = load_documents()
N, V = documents.shape
K = 10
ETA = 100 / V
ALPHA = 1 / K
LAMBDA, GAMMA, PHI = init_variational_params(documents, K)

ELBOs = []
prev_ELBO = -np.inf
curr_ELBO = compute_ELBO(LAMBDA, GAMMA, PHI, documents, nonzero_idxs, K)
ELBOs.append(curr_ELBO)
print(f"Initial ELBO: {ELBOs[0]}\n")

max_iterations = 200
tol = 0.1
LAMBDA_t = copy.deepcopy(LAMBDA)
GAMMA_t = copy.deepcopy(GAMMA)
PHI_t = copy.deepcopy(PHI)

for t in range(max_iterations):
    print(f"Iteration {t+1}")
    for i in tqdm(range(N), desc="Updating PHI and GAMMA"):
        document = documents[i]
        nonzero_idx = nonzero_idxs[i]
        GAMMA_i_t = copy.deepcopy(GAMMA_t[i])
        word_idx = 0
        for idx in nonzero_idx:
            log_PHI_ij = np.zeros((K,))
            for k in range(K):
                LAMBDA_k_t = copy.deepcopy(LAMBDA_t[k])
                exp_propto = digamma(GAMMA_i_t[k]) - digamma(np.sum(GAMMA_i_t)) + digamma(LAMBDA_k_t[idx]) - digamma(np.sum(LAMBDA_k_t))
                log_PHI_ij[k] = exp_propto
            PHI_ij = np.exp(log_PHI_ij - log_sum_exp(log_PHI_ij))
            PHI_t[i][word_idx] = PHI_ij
            word_idx += 1
        GAMMA_i_t = np.zeros((K,)) + ALPHA
        for k in range(K):
            GAMMA_i_t[k] += np.sum(document[nonzero_idx] * PHI_t[i][:, k])
        GAMMA_t[i] = GAMMA_i_t

    for k in tqdm(range(K), desc="Updating LAMBDA"):
        LAMBDA_k_t = np.zeros((V,)) + ETA
        for i in range(N):
            document = documents[i]
            nonzero_idx = nonzero_idxs[i]
            word_idx = 0
            for idx in nonzero_idx:
                LAMBDA_k_t[idx] += document[idx] * PHI_t[i][word_idx][k]
                word_idx += 1
            LAMBDA_t[k] = LAMBDA_k_t

    prev_ELBO = curr_ELBO
    curr_ELBO = compute_ELBO(LAMBDA_t, GAMMA_t, PHI_t, documents, nonzero_idxs, K)
    ELBOs.append(curr_ELBO)
    print(f"Current ELBO: {curr_ELBO} | Change in ELBO: {curr_ELBO - prev_ELBO}\n")

    if abs(curr_ELBO - prev_ELBO) < tol:
        break
stop = time.time()

LAMBDA_final = copy.deepcopy(LAMBDA_t)
GAMMA_final = copy.deepcopy(GAMMA_t)
PHI_final = copy.deepcopy(PHI_t)

time_iter = np.linspace(0, float(stop-start), len(ELBOs))
ELBOs = np.asarray(ELBOs)
data = {
    "time_iter": time_iter,
    "ELBO": ELBOs
}
ELBO_per_time_iter = pd.DataFrame(data=data)
ELBO_per_time_iter.to_csv("AP_K_30_V_all.csv", index=False)

word_topic_probs = LAMBDA_final / LAMBDA_final.sum(axis=1, keepdims=True)
top_words = {}
for k in range(word_topic_probs.shape[0]):
    top_idxs = np.argsort(word_topic_probs[k, :])[-20:][::-1]
    top_words[k] = [idx_to_words[v] for v in top_idxs]

formatted_text = "Top 20 Words for Each Topic:\n\n"
for topic, words in top_words.items():
    formatted_text += f"Topic {topic + 1}: "
    formatted_text += ", ".join(words) + "\n\n"

print(formatted_text)