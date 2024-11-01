#!/usr/bin/env python
# coding: utf-8



# In[19]:


import re
import pandas as pd
import numpy as np
import torch
import re, string
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict


# ## Q1

# In[20]:


def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        corpus = file.read()
    return corpus


# In[21]:


corpus = read_text('HP1.txt')


# In[22]:


print(corpus[:1160])


# In[23]:


len(corpus)


# ## Q2

# In[24]:


def preprocess_text(text):
    
    
    text = re.sub(r'\\x[0-9a-fA-F]{2}', '', text)
    
    text=''.join([char for char in text if char.isalpha() or char.isspace()])
    text = re.sub(r'\n', ' ', text)
    text = text.lower()
    # Remove punctuation
    additional_punctuation = '—“”‘’'
    text = text.translate(str.maketrans('', '', string.punctuation + additional_punctuation))
    words = text.split()
    return words[:5000]


# In[25]:


words= preprocess_text(corpus)


# In[26]:


words


# In[27]:


len(words)


# In[28]:


from collections import Counter


words = preprocess_text(corpus)


word_counts = Counter(words)


most_common_words = word_counts.most_common()  


for word, count in most_common_words:
    print(f"{word}: {count}")


# ## Q3,4,5

# In[29]:


# unique_words=pd.unique(words)
unique_words = list(set(words))


# In[30]:


unique_words = []
known_words = set()
for word in words:
    if word not in known_words:
        known_words.add(word)
        unique_words.append(word)


# In[31]:


len(unique_words)


# In[32]:


vocab_size = len(unique_words)
word_to_idx = {word: i for i, word in enumerate(unique_words)}
idx_to_word = {i: word for i, word in enumerate(unique_words)}


# In[33]:


def one_hot_vector(word):
    one_hot = np.zeros(vocab_size)
    one_hot[word_to_idx[word]] = 1
    return one_hot


# In[34]:


window_size_2 = 2
window_size_4 = 4

dataset = []

def create_dataset(words, window_size):
    for i, word in enumerate(words):
        for j in range(1, window_size + 1):
            if i - j >= 0:
                dataset.append((one_hot_vector(word), one_hot_vector(words[i - j])))
            if i + j < len(words):
                dataset.append((one_hot_vector(word), one_hot_vector(words[i + j])))
    return dataset

 

#Datasets for different window sizes
dataset_2 = create_dataset(words, window_size_2)
dataset_4 = create_dataset(words, window_size_4)


# In[35]:


dataset_2[:2]


# In[36]:


dataset_4[:2]


# In[37]:


def prepare_training_data(dataset):
    X = []
    Y = []
    for i, j in dataset:
        X.append(i)
        Y.append(j)
    return np.array(X), np.array(Y)

X_2, Y_2 = prepare_training_data(dataset_2)
X_4, Y_4 = prepare_training_data(dataset_4)


# In[38]:


len(X_2), len(Y_2)


# In[39]:


len(X_4), len(Y_4)


# ## Q6

# In[46]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomNormal


initializer = RandomNormal(mean=0.0, stddev=0.05)
learning_rate = 0.01
embed_size = 100

def build_model(vocab_size, embed_size, learning_rate):
    model = Sequential([
        Dense(embed_size, input_shape=(vocab_size,), activation='linear', kernel_initializer=initializer),
        Dense(vocab_size, activation='softmax', kernel_initializer=initializer)
    ])
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model



# Model for window size 2
model_2 = build_model(vocab_size, embed_size, learning_rate)
# model_2.fit(X_2, Y_2, epochs=10, batch_size=256, verbose=True)
history_2 = model_2.fit(X_2, Y_2, epochs=10, batch_size=256, verbose=True)

# Model for window size 4
model_4 = build_model(vocab_size, embed_size, learning_rate)
# model_4.fit(X_4, Y_4, epochs=10, batch_size=256, verbose=True)
history_4 = model_4.fit(X_4, Y_4, epochs=10, batch_size=256, verbose=True)


import matplotlib.pyplot as plt

# Plotting the training loss for window size 2 and 4
def compare_window_sizes(history_2, history_4):
    plt.figure(figsize=(10, 6))
    plt.plot(history_2.history['loss'], label='Window Size 2', linestyle='-', marker='o', color='blue')
    plt.plot(history_4.history['loss'], label='Window Size 4', linestyle='--', marker='x', color='red')
    plt.title('Model Loss per Epoch for Different Window Sizes')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

compare_window_sizes(history_2, history_4)


# ## Q7

# # Model_2

# In[49]:


embedding_weights = model_2.layers[0].get_weights()[0]

def infer_embedding(one_hot_vector):
    
    embedding = np.dot(one_hot_vector, embedding_weights)
    return embedding


word = "harry"  
one_hot_vector = np.zeros(vocab_size)
one_hot_vector[word_to_idx[word]] = 1


embedding = infer_embedding(one_hot_vector)
print("Embedding for the word '{}':".format(word))
print(embedding)


# In[50]:


from sklearn.manifold import TSNE

embeddings_to_visualize = np.array([infer_embedding(np.eye(vocab_size)[word_to_idx[word]]) for word in unique_words])

# Perform t-SNE on the set of embeddings
reduced_embeddings = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(embeddings_to_visualize)

# Plot the t-SNE reduced embeddings
plt.figure(figsize=(10, 10))
for i, word in enumerate(unique_words):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12)
plt.title('2D t-SNE of Word Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


# In[51]:


from sklearn.metrics.pairwise import cosine_similarity


def get_similar_words(input_word, word_to_idx, embedding_weights, top_n=10):
    if input_word not in word_to_idx:
        return []

    input_idx = word_to_idx[input_word]
    input_embedding = embedding_weights[input_idx].reshape(1, -1)
    
    similarities = {}
    for word, idx in word_to_idx.items():
        if word == input_word:
            continue
        word_embedding = embedding_weights[idx].reshape(1, -1)
        similarity = cosine_similarity(input_embedding, word_embedding)[0][0]
        similarities[word] = similarity

    similar_words = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    return similar_words

similar_words = get_similar_words('harry', word_to_idx, embedding_weights, top_n=10)
print(similar_words)


# # Model_4

# In[52]:


import numpy as np


embedding_weightz = model_4.layers[0].get_weights()[0]

def infer_embedding_2(one_hot_vector):
    embeddingz = np.dot(one_hot_vector, embedding_weightz)
    return embeddingz


word = "harry"  
one_hot_vector = np.zeros(vocab_size)
one_hot_vector[word_to_idx[word]] = 1

embedding = infer_embedding_2(one_hot_vector)
print("Embedding for the word '{}':".format(word))
print(embedding)


# In[53]:


embeddings_to_visualize_2 = np.array([infer_embedding_2(np.eye(vocab_size)[word_to_idx[word]]) for word in unique_words])

reduced_embeddings = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(embeddings_to_visualize_2)

plt.figure(figsize=(10, 10))
for i, word in enumerate(unique_words):
    plt.scatter(reduced_embeddings[i, 0], reduced_embeddings[i, 1])
    plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=12)
plt.title('2D t-SNE of Word Embeddings_4')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()


# In[52]:


def get_similar_words(input_word, word_to_idx, embedding_weightz, top_n=10):
    if input_word not in word_to_idx:
        return []

   
    input_idx = word_to_idx[input_word]
    input_embedding = embedding_weights[input_idx].reshape(1, -1)
    
    similarities = {}
    for word, idx in word_to_idx.items():
        if word == input_word:
            continue
        word_embedding = embedding_weightz[idx].reshape(1, -1)
        similarity = cosine_similarity(input_embedding, word_embedding)[0][0]
        similarities[word] = similarity

    similar_wordz = sorted(similarities, key=similarities.get, reverse=True)[:top_n]
    return similar_wordz

similar_wordz = get_similar_words('harry', word_to_idx, embedding_weightz, top_n=10)
print(similar_wordz)


# In[ ]:





# In[ ]:




