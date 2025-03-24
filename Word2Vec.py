import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk import download
from nltk.corpus import reuters
from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split

download('reuters')

from datasets import load_dataset

ds = load_dataset("allmalab/aze-books")
docs = list(ds["train"]["text"])[:]
vectorizer = CountVectorizer(min_df = 10)
X_term_doc = vectorizer.fit_transform(docs)

# Statistics
vocab = vectorizer.get_feature_names_out()
word_counts = np.asarray(X_term_doc.sum(axis=0)).flatten()

print(f'Dataset size: {len(docs)} documents')
print(f'Number of distinct words: {len(vocab)}')

freq_threshold = 20
frequent_words = np.sum(word_counts >= freq_threshold)
rare_words = np.sum(word_counts < freq_threshold)

print(f'Frequent words (≥ {freq_threshold} occurrences): {frequent_words}')
print(f'Rare words (< {freq_threshold} occurrences): {rare_words}')

# Task 2: Word2Vec
sentences = [re.findall(r'\b\w+\b', doc.lower()) for doc in docs]
w2v_model = Word2Vec(sentences, vector_size=16, window=5, min_count=10, workers=4)

w2v_model.save("word2vec3.model")
