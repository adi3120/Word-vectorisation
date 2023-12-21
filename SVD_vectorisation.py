import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import string
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def co_occurrence_matrix(corpus, window_size):
    corpus = corpus.translate(str.maketrans('', '', string.punctuation))
    corpus = corpus.lower()
    corpus_list = word_tokenize(corpus)
    
    stop_words = set(stopwords.words('english'))
    filtered_corpus = [word for word in corpus_list if word not in stop_words]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = [lemmatizer.lemmatize(word) for word in filtered_corpus]
    
    vocabulary = list(set(lemmatized_corpus))
    
    matrix = pd.DataFrame(0, index=vocabulary, columns=vocabulary)
    
    for i, word in enumerate(lemmatized_corpus):
        if i + window_size < len(lemmatized_corpus):
            context = lemmatized_corpus[i+1:i+window_size+1]
            for ctx in context:
                matrix.at[word, ctx] += 1
    
    np.fill_diagonal(matrix.values, 0)
    return matrix,vocabulary

def ppmi(matrix):
    total_sum = np.sum(matrix.values)
    row_sum = np.sum(matrix, axis=1)
    col_sum = np.sum(matrix, axis=0)
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            pmi = max(np.log2((val * total_sum) / (row_sum[i] * col_sum[j])), 0)
            matrix.iloc[i, j] = pmi
    return matrix

corpus = """Dogs are wonderful companions . They enjoy playing fetch and running in the park. A well-trained dog can learn many tricks. Cats are independent creatures, often preferring solitude. Their agility and grace are admired by many. Some people love both dogs and cats equally, while others have a strong preference for one over the other. Dogs require regular exercise, whereas cats are more low-maintenance. Dog owners often take their pets for walks, while cats enjoy lounging indoors. The debate between dog lovers and cat enthusiasts is never-ending, each having valid reasons for their preferences. Nevertheless, both dogs and cats bring joy and comfort to countless households.
"""

co_matrix,vocabulary = co_occurrence_matrix(corpus, 2)
ppmi_matrix = ppmi(co_matrix)
ppmi_matrix = ppmi_matrix.replace([np.inf, -np.inf], np.nan).fillna(0)

sparse_matrix = csc_matrix(ppmi_matrix.values.astype(float))

# Perform SVD with scipy truncated SVD
k = 2  # Number of components for SVD
U, S, VT = svds(sparse_matrix, k=k)

# word_embeddings = U[:, ::-1]  # Rearrange for compatibility with t-SNE
word_embeddings = U @ np.diag(S) @ VT

# t-SNE visualization (same as before)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(word_embeddings)

# Scatter plot visualization (same as before)
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

for i, word in enumerate(vocabulary):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization of Word Embeddings')
plt.show()
