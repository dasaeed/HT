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

class LDA:
    def __init__(self, n_topics: int, ALPHA: float, ETA: float):
        self.K = n_topics
        self.ALPHA = ALPHA
        self.ETA = ETA

    def generate_data(self, vocab_size: int, n_docs: int, doc_lengths: List[int]) -> Tuple[np.ndarray, List[np.ndarray]]:
        self.V = vocab_size
        BETA = np.random.dirichlet(np.full(self.V, self.ETA), size=self.K)
        docs = []
        for doc_len in doc_lengths:
            THETA = np.random.dirichlet(np.full(self.K, self.ALPHA))
            doc = []
            for _ in range(doc_len):
                z = np.random.choice(self.K, p=THETA)
                w = np.random.choice(self.V, p=BETA[z])
                doc.append(w)
            docs.append(np.array(doc))
        return BETA, docs
