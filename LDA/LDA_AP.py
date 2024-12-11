import os
import time
import pandas as pd
import numpy as np
import numpy.random as npr
import re
import nltk
import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from scipy.special import digamma, loggamma

with open("vocab.txt", "r") as file:
    vocab = [line.strip() for line in file]
vocab_to_idx = {word: idx for idx, word in enumerate(vocab)}

with open("ap.txt", "r") as file:
    raw_text = file.read()
documents = re.findall(r"<TEXT>(.*?)</TEXT>", raw_text, re.DOTALL)
documents = [doc.strip().replace("\n", " ") for doc in documents if doc.strip()]

stop_words = set(stopwords.words("english"))
corpus_matrix = []
for doc in documents:
    tokens = word_tokenize(doc.lower())
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    word_idxs = [vocab_to_idx[word] for word in filtered_tokens if word in vocab_to_idx]
    if word_idxs:
        corpus_matrix.append(word_idxs)

K = 30
V = len(vocab)
ALPHA = 0.5
ETA = 1 / V

def log_sum_exp(vec):
    a = np.max(vec, axis=0)
    log_sum_exp = np.log(np.sum(np.exp(vec - a))) + a

    return log_sum_exp

def init_variational_params(corpus_mat, K, V):
    N = len(corpus_mat)
    LAMBDA = np.random.uniform(low=0.01, high=1.00, size=(K, V))
    GAMMA = np.ones((N, K))
    PHI = [np.ones((len(doc), K)) / K for doc in corpus_mat]

    return LAMBDA, GAMMA, PHI

def compute_ELBO(LAMBDA, GAMMA, PHI, corpus_mat, ALPHA=ALPHA, ETA=ETA):
    E_log_p_BETA = np.sum((ETA - 1) * (digamma(LAMBDA) - digamma(np.sum(LAMBDA, axis=1, keepdims=True))))

    E_log_p_THETA = np.sum((ALPHA - 1) * (digamma(GAMMA) - digamma(np.sum(GAMMA, axis=1, keepdims=True))))

    E_log_p_z_x = 0
    for i, doc in enumerate(corpus_mat):
        for j, x_ij in enumerate(doc):
            E_log_p_z_x += np.sum(
                PHI[i][j] * (
                    digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i])) +
                    digamma(LAMBDA[:, x_ij]) - digamma(np.sum(LAMBDA, axis=1))
                )
            )

    E_q_log_BETA = np.sum(
        -loggamma(np.sum(LAMBDA, axis=1)) + np.sum(loggamma(LAMBDA), axis=1) -
        np.sum((LAMBDA - 1) * (digamma(LAMBDA) - digamma(np.sum(LAMBDA, axis=1, keepdims=True))), axis=1)
    )

    E_q_log_THETA = np.sum(
        -loggamma(np.sum(GAMMA, axis=1)) + np.sum(loggamma(GAMMA), axis=1) -
        np.sum((GAMMA - 1) * (digamma(GAMMA) - digamma(np.sum(GAMMA, axis=1, keepdims=True))), axis=1)
    )

    E_log_q_z = 0
    for i, doc in enumerate(corpus_mat):
        for j, x_ij in enumerate(doc):
            E_log_q_z += -np.sum(PHI[i][j] * np.log(PHI[i][j] + 1e-10))
            
    ELBO = E_log_p_BETA + E_log_p_THETA + E_log_p_z_x + E_q_log_BETA + E_q_log_THETA + E_log_q_z

    return ELBO

def update_variational_params(LAMBDA, GAMMA, PHI, corpus_mat, K, V):
    N= len(corpus_mat)
    for i in tqdm(range(N), desc="Updating PHI and GAMMA"):
    # for i in range(N):
        M = len(corpus_mat[i])
        for j in range(M):
            x_ij = corpus_mat[i][j]
            exp_propto = digamma(LAMBDA[:, x_ij]) - digamma(np.sum(LAMBDA[:, x_ij])) \
                + digamma(GAMMA[i]) - digamma(np.sum(GAMMA[i]))
            PHI[i][j] = np.exp(exp_propto - log_sum_exp(exp_propto))

        for k in range(K):
            GAMMA[i, k] = ALPHA + np.sum(PHI[i][:, k])

    LAMBDA = np.full((K, V), ETA)
    for i, doc in enumerate(tqdm(corpus_mat, desc="Updating LAMBDA")):
    # for i, doc in enumerate(corpus_mat):
        doc_word_idxs = np.array(doc)
        for k in range(K):
            np.add.at(LAMBDA[k], doc_word_idxs, PHI[i][:, k])

    return LAMBDA, GAMMA, PHI

tol = 1e-2
iteration = 1
max_iterations = 1000
curr_ELBO = 0
prev_ELBO = 301
ELBOs = []

start = time.time()
LAMBDA, GAMMA, PHI = init_variational_params(corpus_mat=corpus_matrix, K=K, V=V)
ELBOs.append(compute_ELBO(LAMBDA, GAMMA, PHI, corpus_mat=corpus_matrix))
print(f"Initial ELBO: {ELBOs[0]}\n")
while iteration <= max_iterations:
    print(f"Iteration: {iteration}")
    LAMBDA, GAMMA, PHI = update_variational_params(LAMBDA, GAMMA, PHI, corpus_mat=corpus_matrix, K=K, V=V)
    prev_ELBO = curr_ELBO
    curr_ELBO = compute_ELBO(LAMBDA, GAMMA, PHI, corpus_mat=corpus_matrix)
    print(f"Current ELBO: {curr_ELBO}")
    ELBOs.append(curr_ELBO)

    if np.abs(curr_ELBO - prev_ELBO) <= tol:
        break

    iteration += 1
    print("\n")

stop = time.time()
plt.xlabel("Seconds")
plt.ylabel("ELBO")
plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
plt.plot(np.linspace(0, float(stop-start), len(ELBOs)), ELBOs)

time_iter = np.linspace(0, float(stop-start), len(ELBOs))
ELBOs = np.asarray(ELBOs)
data = {
    "time_iter": time_iter,
    "ELBO": ELBOs
}
ELBO_per_time_iter = pd.DataFrame(data=data)
ELBO_per_time_iter.to_csv("ELBOs_full_vocab.csv", index=False)

word_topic_probs = LAMBDA / LAMBDA.sum(axis=1, keepdims=True)
top_words = {}
for k in range(word_topic_probs.shape[0]):
    top_idxs = np.argsort(word_topic_probs[k, :])[-10:][::-1]
    top_words[k] = [vocab[v] for v in top_idxs]

formatted_text = "Top 10 Words for Each Topic:\n\n"
for topic, words in top_words.items():
    formatted_text += f"Topic {topic + 1}: "
    formatted_text += ", ".join(words) + "\n\n"

print(formatted_text)