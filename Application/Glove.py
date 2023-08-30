
from numpy import zeros, dtype, float32 as REAL, ascontiguousarray, fromstring
from gensim import utils
import gensim
import numpy as np
import matplotlib as mp
with open("tokenization/custom_tokenizer/tokenized_data.txt",mode="r",encoding='utf-8') as file:
    tdt = file.readlines()

# print(type(tdt))
tokens = []
for i in tdt:
  tokens.extend(i.split(" "))
# print((tokens[0]))
temp = []
for i in tokens:
    temp.append(i.replace('"', '').replace(',', ''))
# print(temp[0])
from collections import Counter, defaultdict

# # Returns a dictionary `w -> freq`, mapping word strings to word corpus frequency.
def build_vocab(tokens_en):
    vocab = {}
    for token in tokens_en:
        vocab[token] = vocab.get(token, 0) + 1

    return vocab

# # Build a word co-occurrence list for the given corpus as described in Pennington et al.(2014)
def build_cooccur(vocab, tokens_en, window_size):
    cooccurence_matrix = defaultdict(lambda: 0)
    vocab_size = len(vocab)

    for i, word in enumerate(tokens_en):
      for j in range(max(0, i - window_size), min(len(tokens_en), i + window_size + 1)):
        if j != i and word in vocab and tokens_en[j] in vocab:
          cooccurence_matrix[word, tokens_en[j]] = cooccurence_matrix.get((tokens_en[i], tokens_en[j]), 0) + 1

    return cooccurence_matrix


from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import random
nltk.download('punkt')  # in order to make "nltk.tokenize.word_tokenize" work

tokens_en = temp
vocab = build_vocab(tokens_en)              
sorted_vocab = sorted(vocab.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)
sorted_vocab = sorted_vocab[:min(200, len(sorted_vocab))]                 # limits the vocab to a size of 10,000
vocab = {word[0] : word[1] for word in sorted_vocab}
cooccurence_matrix = build_cooccur(vocab, tokens_en, window_size = 4)       # window size can be adjusted, taken as 4 here
print("Length of Vocab is :", len(vocab))
print("Length of cooccurence_matrix is :", len(cooccurence_matrix))
# print(cooccurence_matrix)


import numpy as np

# function f(x) which is used to calculate cost
def f(x, x_max, alpha):
  if x < x_max:
    return (x / x_max) ** alpha
  else:
    return 1

# does one step of the gradient descent algorithm
def CostUpd_iter(vocab, cooccurence_matrix, W_center, U_context, bias_center, bias_context, learning_rate, x_max, alpha):
  
  # initialize the gradient vectors and cost by 0
  cost = 0
  dW_center = np.zeros(W_center.shape, dtype = "float64")
  dU_context = np.zeros(U_context.shape, dtype = "float64")
  dbias_center = np.zeros(bias_center.shape, dtype = "float64")
  dbias_context = np.zeros(bias_context.shape, dtype = "float64")

  # for every main word, all the words from vocab are considered for calculation of gradients
  # refer to report for mathematical derivations of these derivatives
  for i, word1 in enumerate(vocab.keys()):
    for j, word2 in enumerate(vocab.keys()):
      x = cooccurence_matrix.get((word1, word2), 0)
      f_x_ij = f(x, x_max, alpha)
      gradient1 = 0
      gradient2 = 0

      if x > 0:
        gradient1 = np.dot(W_center[i], U_context[j]) + bias_center[i] + bias_context[j] - np.log(x)
        gradient2 = np.dot(W_center[j], U_context[i]) + bias_center[j] + bias_context[i] - np.log(x)
        cost += f_x_ij * gradient1 * gradient1

      dW_center[i] += 2 * f_x_ij * gradient1 * U_context[j]
      dbias_center[i] += 2 * f_x_ij * gradient1
      dU_context[i] += 2 * f_x_ij * gradient2 * W_center[j]
      dbias_context[i] += 2 * f_x_ij * gradient2

  # update the gradient vectors (W -> W - α * dW)
  W_center = W_center - learning_rate * dW_center
  bias_center = bias_center - learning_rate * dbias_center
  U_context = U_context - learning_rate * dU_context
  bias_context = bias_context - learning_rate * dbias_context

  return cost, W_center, U_context, bias_center, bias_context


# main function which is called for training the corpus to calculate word embeddings
def train_glove(vocab, cooccurence_matrix, vector_size, iterations, alpha, x_max, learning_rate):
    vocab_size = len(vocab)
    costlist=[]
    # Initialize 2 random word vector matrices in range (-0.5, 0.5] for each token,
    # one for the token as (main) center and one for the token as context 
    W_center = (np.random.rand(vocab_size, vector_size))
    U_context = (np.random.rand(vocab_size, vector_size))

    # Bias terms, each associated with a single vector.
    bias_center = (np.random.rand(vocab_size))
    bias_context = (np.random.rand(vocab_size))

    total_cost = 0

    # prints the cost after each epoch of the training set, learning rate should be such that cost decrease
    for i in range(iterations):
      cost, W_center, U_context, bias_center, bias_context = CostUpd_iter(vocab, cooccurence_matrix, W_center, U_context, bias_center, bias_context, learning_rate, x_max, alpha)
      print("Cost after Iteration ", i)
      print(" =", cost)
      total_cost = cost
      costlist.append(cost)

    return W_center, U_context, bias_center, bias_context, total_cost, iterations,costlist


W_center, U_context, bias_center, bias_context, total_cost, iterations,costlist = train_glove(vocab, cooccurence_matrix, vector_size = 100, iterations = 35, alpha = 0.75, x_max = 100, learning_rate = 0.0005)\

# storing the word embeddings in a dictionary
import random
emb = {}
for i, word in enumerate(vocab.keys()):
  emb[word] = W_center[i]

print(type(emb['Java']))
m = gensim.models.keyedvectors.Word2VecKeyedVectors(vector_size=100)
for i in emb:
  # print(len(emb[i]))
  m.__setitem__(i,emb[i])
# print(costlist)
