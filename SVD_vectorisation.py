import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import pandas as pd
import math
from numpy.linalg import eig
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

stop_words=["i","me","my","myself","we","our","ours","ourselves","you","your","yours","yourself","yourselves","he","him","his","himself","she","her","hers","herself","it","its","itself","they","them","their","theirs","themselves","what","which","who","whom","this","that","these","those","am","is","are","was","were","be","been","being","have","has","had","having","do","does","did","doing","a","an","the","and","but","if","or","because","as","until","while","of","at","by","for","with","about","against","between","into","through","during","before","after","above","below","to","from","up","down","in","out","on","off","over","under","again","further","then","once","here","there","when","where","why","how","all","any","both","each","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","s","t","can","will","just","don","should","now"]

def corpus_remove_punc_symb(corpus):
  punctuations='''!()-[]{};:'"\,<>./?@#$%^&*~'''
  no_punct=""
  for char in corpus:
    if char not in punctuations:
      no_punct+=char
  return no_punct

def corpus_tokenise(corpus):
  tokens=corpus.split()
  return tokens

def corpus_remove_stop_words(corpus):
  coList=corpus_tokenise(corpus)
  for word in coList:
    if word in stop_words:
      coList.remove(word)
  return " ".join(coList)

def corpus_lemmatize(corpus):
  lemmatizer = WordNetLemmatizer()
  tokens=corpus_tokenise(corpus)
  for i in range(0,len(tokens)):
    tokens[i]=lemmatizer.lemmatize(tokens[i])
  return " ".join(tokens)


def corpus_prepare(corpus):
  corpus=corpus.lower()
  corpus=corpus_remove_punc_symb(corpus)
  corpus=corpus_remove_stop_words(corpus)
  corpus=corpus_lemmatize(corpus)
  return corpus

def vocabulary_prepare(corpus):
  vocabulary=list(set(corpus_tokenise(corpus)))
  return vocabulary

def co_matrix_prepare(corpus,vocabulary,window_size):
  matrix={}
  context={}
  for i in range(0,len(vocabulary)):
    context[vocabulary[i]]=0

  for i in range(0,len(vocabulary)):
    matrix[vocabulary[i]]=context

  matrix = pd.DataFrame.from_dict(matrix)
  corpus_list=corpus_tokenise(corpus)

  for i in range(0,len(corpus_list)):
    windowwords=[]
    if i+window_size<len(corpus_list):
      for j in range(0,window_size):
        windowwords.append(corpus_list[i+j])
        matrix[corpus_list[i]][corpus_list[i+j]]+=1
    else:
      break

  for i in matrix.keys():
    matrix[i][i]=0

  return matrix

def countOccurrences(str, word):

    wordslist = list(str.split())
    return wordslist.count(word)

def PMI(count_wc,count_c,count_w,N):
  return math.log((count_wc*N)/(count_c*count_w))

def PPMI(count_wc,count_c,count_w,N,eps=1e-6):
  if count_wc*N!=0 and count_c*count_w!=0:
    return max(eps,math.log((count_wc*N)/(count_c*count_w)))
  else:
    return 0

def PMI_co_matrix(matrix,corpus,ppmi=True):
  for word in matrix.keys():
    for context in matrix[word].keys():
      if ppmi==False:
        matrix[word][context] = PMI(matrix[word][context],countOccurrences(corpus,context),countOccurrences(corpus,word),len(matrix.keys()))
      else:
        matrix[word][context] = PPMI(matrix[word][context],countOccurrences(corpus,context),countOccurrences(corpus,word),len(matrix.keys()))
  return matrix

def SVD(A):
    # Step 1: Compute A^TA
    ATA = np.dot(A.T, A)
    
    # Step 2: Calculate eigenvalues and eigenvectors of A^TA
    eigenvalues, V = np.linalg.eig(ATA)
    
    # Sort eigenvalues in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    V = V[:, idx]

    
    # Construct Sigma matrix
    Sigma = np.sqrt(np.diag(eigenvalues))
    
    # Step 3: Calculate U matrix
    U = np.zeros((A.shape[0], len(eigenvalues)))
    for i in range(len(eigenvalues)):
        if eigenvalues[i] > 0:
            U[:, i] = (1 / np.sqrt(eigenvalues[i])) * np.dot(A, V[:, i])
    
    return U, Sigma, V

def rank_k_approximation(A, k):
    # Compute SVD
    U, Sigma, V = SVD(A)
    
    # Rank-k approximation
    approx_U = U[:, :k]
    approx_Sigma = Sigma[:k, :k]
    approx_Vt = V.T[:k, :]
    
    
    return approx_U,approx_Sigma,approx_Vt

def co_matrix_rank_k_approximation(co_matrix,k):
  co_matrix_np = co_matrix.to_numpy()
  approx_U,approx_Sigma,approx_Vt = rank_k_approximation(co_matrix_np, k)
  rank_k_approx=approx_U @ approx_Sigma @ approx_Vt
  rank_k_approx_df = pd.DataFrame(rank_k_approx, index=co_matrix.index, columns=co_matrix.columns)
  return rank_k_approx_df

def cosine_similarity(co_matrix):
  co_matrix_np = co_matrix.to_numpy()
  XXT=np.dot(co_matrix,co_matrix.T)
  XXT = pd.DataFrame(XXT, index=co_matrix.index, columns=co_matrix.columns)
  return XXT

def co_matrix_rank_k_SVD_word_context_embeddings(co_matrix,k):
  co_matrix_np = co_matrix.to_numpy()
  approx_U,approx_Sigma,approx_Vt = rank_k_approximation(co_matrix_np, k)
  word_matrix=approx_U @ approx_Sigma
  word_matrix = pd.DataFrame(word_matrix, index=co_matrix.index)
  context_matrix=pd.DataFrame(approx_Vt.T,index=co_matrix.index)
  return word_matrix,context_matrix

corpus = """Dogs are wonderful companions . 
They enjoy playing fetch and running in the park.
 A well-trained dog can learn many tricks. 
 Cats are independent creatures, often preferring solitude. 
 Their agility and grace are admired by many. 
 Some people love both dogs and cats equally, while others have a strong preference for one over the other. 
 Dogs require regular exercise, whereas cats are more low-maintenance. 
 Dog owners often take their pets for walks, while cats enjoy lounging indoors. 
 The debate between dog lovers and cat enthusiasts is never-ending, each having valid reasons for their preferences. 
 Nevertheless, both dogs and cats bring joy and comfort to countless households.
"""

corpus=corpus_prepare(corpus)
vocabulary=vocabulary_prepare(corpus)
co_matrix=co_matrix_prepare(corpus,vocabulary,3)
co_matrix=PMI_co_matrix(co_matrix,corpus)

word_embeddings,context_embeddings=co_matrix_rank_k_SVD_word_context_embeddings(co_matrix,20)

word_embeddings

context_embeddings

def find_closest_words(word_embeddings, target_word, n=10):
    if target_word not in word_embeddings.index:
        print("Word not found in the vocabulary.")
        return
    
    target_embedding = word_embeddings.loc[target_word]
    similarities = word_embeddings.dot(target_embedding)  # Calculate cosine similarity
    
    # Sort similarities and get the top n+1 (excluding the input word itself)
    closest_words = similarities.nlargest(n + 1)[1:]
    return closest_words

# Test with a word and get 10 closest words
target_word = 'dog'
closest_words = find_closest_words(word_embeddings, target_word)
print(closest_words)

target_word = 'exercise'
closest_words = find_closest_words(word_embeddings, target_word)
print(closest_words)

