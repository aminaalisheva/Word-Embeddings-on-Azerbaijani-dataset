# Word-Embeddings-on-Azerbaijani-dataset
**Date:** 13.03.2025  
**Contributers:**  
- Amina Alisheva  
- Ariana Kenbayeva  
- Jafar Isbarov  

---

## Introduction  
This project focuses on word embeddings using Azerbaijani datasets. Our primary objectives include:
- Analyzing the dataset and constructing a term-document matrix and word-word matrix.
- Training Word2Vec and GloVe models to find synonyms, test the models, and analyze vector arithmetic patterns.
- Applying Logistic Regression for text classification using various feature extraction methods and comparing their performance.

---

## Methodology  
### Dataset Preparation  
- **Datasets:** Azerbaijani books and dissertation topics.
- **Preprocessing Steps:** Tokenization, stopword removal, spell correction, lemmatization.

### Feature Representation  
- **Term-Document Matrix**: Captures word frequency across documents.
- **Word-Word Matrix**: Analyzes word co-occurrence relationships.

### Word Embeddings  
- **Word2Vec (Skip-Gram Model)**
  - vector_size = 16
  - window = 5
  - min_count = 10
- **GloVe**
  - vector_size = 16
  - window = 5
  - min_count = 10
  - max_iter = 15

### Text Classification  
- **Model:** Logistic Regression with L1 regularization.
- **Feature Extraction Methods:** Count Vectorizer, TF-IDF, PMI, Word2Vec, and GloVe.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score.

---

## Results  
### Dataset Statistics  
- **Documents:** 3,602
- **Unique Words:** 553,395
- **Frequent Words (≥100 occurrences):** 98,097
- **Rare Words (<10 occurrences):** 163,832

### Word Embedding Analysis  
**Word2Vec & GloVe Synonyms Comparison:**  
- Word2Vec produced more domain-specific synonyms.
- GloVe captured broader co-occurrence relationships but sometimes yielded loosely related terms.
- Word2Vec demonstrated superior semantic consistency in vector arithmetic.

### Text Classification Performance  
- Logistic Regression was applied using different embedding-based feature sets.
- TF-IDF and Word2Vec showed better classification performance compared to PMI and GloVe.

---

## Discussion  
- **Challenges:** Handling rare words, computational limitations, polysemy in embeddings.
- **Findings:** Word2Vec outperformed GloVe in domain-specific synonym retrieval.
- **Future Work:** Exploring transformer-based models (e.g., BERT) for improved contextual embeddings.

---

## Contributions  
- **Amina Alisheva:** Implemented Logistic Regression, evaluated classification models.
- **Ariana Kenbayeva:** Handled data preprocessing, created term-document & word-word matrices.
- **Jafar Isbarov:** Trained Word2Vec and GloVe, analyzed embedding performance.
- **All Members:** Contributed to report writing and presentation preparation.

---

## Conclusion  
This project explored word embeddings through various methodologies, revealing insights into their effectiveness for text classification. Future research could focus on improving embeddings using deep learning models and larger datasets.

---

## References  
1. Hajili. (2024). AzSci Topics. Hugging Face. https://huggingface.co/datasets/hajili/azsci_topics  
2. Huseynova, K., Isbarov, J., & Aghalarov, M. (n.d.). Aze-books dataset.  
