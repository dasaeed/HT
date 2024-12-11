import os
import time
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

from typing import List, Dict, Tuple, Set, Optional
import logging

class CAVI_LDA:
    def __init__(self, n_topics: int, ALPHA: float = 0.1, ETA: float = 0.01, 
                 max_iter: int = 100, tol: float = 1e-3):
        self.K = n_topics
        self.ALPHA = ALPHA
        self.ETA = ETA
        self.max_iter = max_iter
        self.tol = tol

        self.n_docs = None
        self.vocab_size = None
        self.LAMBDA: Optional[np.ndarray] = None

    def initialize_variational_parameters(self, X: np.ndarray) -> None:
        self.n_docs, self.vocab_size = X.shape
        self.LAMBDA = np.random.gamma(100., 1./100., (self.K, self.vocab_size))
    
    def update_PHI(self, doc_idx: int, doc: np.ndarray, GAMMA: np.ndarray) -> np.ndarray:
        word_ids = np.nonzero(doc)[0]
        word_counts = doc[word_ids]

        PHI = np.zeros((len(word_ids), self.K))
        E_log_BETA = digamma(self.LAMBDA) - digamma(self.LAMBDA.sum(axis=1))[:, np.newaxis]
        E_log_THETA = digamma(GAMMA) - digamma(GAMMA.sum())
        PHI = np.exp(E_log_THETA[np.newaxis, :] + E_log_BETA[:, word_ids].T)
        PHI = PHI / PHI.sum(axis=1)[:, np.newaxis]

        return PHI, word_ids, word_counts
    
    def update_GAMMA(self, PHI: np.ndarray, word_counts: np.ndarray) -> np.ndarray:
        GAMMA = self.ALPHA + np.dot(word_counts, PHI)

        return GAMMA
    
    def update_LAMBDA(self, X: np.ndarray, PHI_all: List[np.ndarray], 
                      word_ids_all: List[np.ndarray], word_counts_all: List[np.ndarray]) -> None:
        self.LAMBDA = np.full_like(self.LAMBDA, self.ETA)
        for d in range(self.n_docs):
            PHI = PHI_all[d]
            word_ids = word_ids_all[d]
            word_counts = word_counts_all[d]
            self.LAMBDA[:, word_ids] += np.dot(PHI.T, word_counts[:, np.newaxis])
    
    def calculate_ELBO(self, X: np.ndarray, PHI_all: List[np.ndarray], 
                       GAMMA_all: np.ndarray, word_ids_all: List[np.ndarray],
                       word_counts_all: List[np.ndarray]) -> float:
        ELBO = 0.

        E_log_THETA = digamma(GAMMA_all) - digamma(GAMMA_all.sum(axis=1))[:, np.newaxis]
        ELBO += (self.ALPHA - 1) * E_log_THETA.sum()
        ELBO += np.sum(loggamma(GAMMA_all.sum(axis=1)) - loggamma(self.ALPHA * self.K))
        ELBO -= np.sum(loggamma(GAMMA_all) - loggamma(self.ALPHA))

        E_log_BETA = digamma(self.LAMBDA) - digamma(self.LAMBDA.sum(axis=1))[:, np.newaxis]
        ELBO += (self.ETA - 1) * E_log_BETA.sum()
        ELBO += np.sum(loggamma(self.LAMBDA.sum(axis=1)) - loggamma(self.ETA * self.vocab_size))
        ELBO -= np.sum(loggamma(self.LAMBDA) - loggamma(self.ETA))

        for d in range(self.n_docs):
            PHI = PHI_all[d]
            word_ids = word_ids_all[d]
            word_counts = word_counts_all[d]
            ELBO += np.sum(word_counts[:, np.newaxis] * PHI *
                           (E_log_THETA[d] + E_log_BETA[:, word_ids].T - np.log(PHI + 1e-100)))
        
        return ELBO
    
    def fit(self, X: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        self.initialize_variational_parameters(X)
        GAMMA_all = np.full((self.n_docs, self.K), self.ALPHA + X.sum(axis=1).mean() / self.K)
        PHI_all = []
        word_ids_all = []
        word_counts_all = []
        prev_ELBO = -10000

        for iteration in range(self.max_iter):
            PHI_all = []
            word_ids_all = []
            word_counts_all = []

            for d in range(self.n_docs):
                PHI, word_ids, word_counts = self.update_PHI(d, X[d], GAMMA_all[d])
                PHI_all.append(PHI)
                word_ids_all.append(word_ids)
                word_counts_all.append(word_counts)
                GAMMA_all[d] = self.update_GAMMA(PHI, word_counts)

            self.update_LAMBDA(X, PHI_all, word_ids_all, word_counts_all)

            ELBO = self.calculate_ELBO(X, PHI_all, GAMMA_all, word_ids_all, word_counts_all)
            delta_ELBO = (ELBO - prev_ELBO) / abs(prev_ELBO)
            logging.info(f"Iteration {iteration}: ELBO = {ELBO:.2f} (delta = {delta_ELBO:.4f})")

            if iteration > 0 and delta_ELBO < self.tol:
                logging.info("Converged")
                break
            prev_ELBO = ELBO

        return PHI_all, GAMMA_all
    
    def get_topics(self, n_top_words: int = 10) -> List[List[int]]:
        return [
            list(self.LAMBDA[k].argsort()[-n_top_words:][::-1]) for k in range(self.K)
        ]
    
def load_vocabulary(vocab_file: str) -> Dict[str, int]:
    stop_words = set(stopwords.words("english"))
    vocab = {}
    with open(vocab_file, "r") as file:
        for idx, word in enumerate(file):
            if word not in stop_words:
                vocab[word.strip()] = idx

    return vocab

def preprocess_text(text: str) -> List[str]:
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    words = re.findall(r"\b\w+\b", text)
    words = [word for word in words if word not in stop_words]

    return words

def parse_documents(doc_file: str) -> List[str]:
    with open(doc_file, "r") as file:
        raw_text = file.read()
    
    pattern = r"<TEXT>\s*(.*?)\s*</TEXT>"
    documents = re.findall(pattern, raw_text, re.DOTALL)

    return documents

def create_document_term_matrix(documents: List[str], vocab: Dict[str, int]) -> np.ndarray:
    n_docs = len(documents)
    vocab_size = len(vocab)
    doc_term_matrix = np.zeros((n_docs, vocab_size), dtype=int)

    for doc_idx, doc in enumerate(documents):
        words = preprocess_text(doc)
        for word in words:
            if word in vocab:
                vocab_idx = vocab[word]
                doc_term_matrix[doc_idx, vocab_idx] += 1
    
    return doc_term_matrix

def run_LDA_analysis(doc_file: str, vocab_file: str, n_topics: int = 25, 
                     ALPHA: float = 0.1, ETA: float = 0.01) -> Tuple[CAVI_LDA, List[str]]:
    vocab = load_vocabulary(vocab_file)
    vocab_list = [""] * len(vocab)
    for word, idx in vocab.items():
        vocab_list[idx] = word

    logging.info("Parsing documents...")
    documents = parse_documents(doc_file)

    logging.info("Creating document-term matrix...")
    dtm = create_document_term_matrix(documents, vocab)

    logging.info("Fitting LDA model...")
    model = CAVI_LDA(
        n_topics=n_topics, ALPHA=ALPHA, ETA=ETA
    )
    model.fit(dtm)

    return model, vocab_list

def print_topics(model: CAVI_LDA, vocab_list: List[str], n_words: int = 10):
    topic_words = model.get_topics(n_words)
    for topic_idx, word_indices in enumerate(topic_words):
        words = [vocab_list[idx] for idx in word_indices]
        print(f"\nTopic {topic_idx + 1}:")
        print(", ".join(words))

if __name__=="__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )       
    model, vocab_list = run_LDA_analysis(
        doc_file="ap.txt",
        vocab_file="vocab.txt",
        n_topics=25
    )
    print("\nDiscovered Topics:")
    print_topics(model, vocab_list)
