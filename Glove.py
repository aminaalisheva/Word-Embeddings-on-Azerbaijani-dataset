import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from nltk import download
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec, KeyedVectors
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from glove import Corpus, Glove

download('punkt')
download('reuters')


ds = load_dataset("allmalab/aze-books")
docs = list(ds["train"]["text"])[:]
vectorizer = CountVectorizer(min_df = 10)
X_term_doc = vectorizer.fit_transform(docs)

# Tokenize sentences
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in docs]

# Prepare the corpus
corpus = Corpus()
corpus.fit(tokenized_sentences, window=5)

# Initialize and train GloVe model
glove = Glove(no_components=50, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

# Save embeddings to file
glove.save('glove.model')

# Example usage: getting embedding for a word
print(glove.word_vectors[glove.dictionary['sentence']])
