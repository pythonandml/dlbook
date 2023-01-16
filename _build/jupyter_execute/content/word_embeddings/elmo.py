#!/usr/bin/env python
# coding: utf-8

# # 4.3.1. Embeddings from Language Models (ELMo)
# 
# ELMo is an NLP framework developed by [AllenNLP](https://allenai.org/allennlp/software/elmo). ELMo word vectors are calculated using a two-layer bidirectional language model (biLM). Each layer comprises forward and backward pass.
# 
# Unlike traditional word embeddings such as [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html) (link to previous chapter), [GloVe](https://pythonandml.github.io/dlbook/content/word_embeddings/glove.html) (link to previous chapter) or [FastText](https://pythonandml.github.io/dlbook/content/word_embeddings/fasttext.html) (link to previous chapter), the ELMo vector assigned to a token or word is actually a function of the entire sentence containing that word. Therefore, the same word can have different word vectors under different contexts as explained in [Contextual Word embeddings](https://pythonandml.github.io/dlbook/content/word_embeddings/contextual_word_embeddings.html) (link to previous chapter).
# 
# ### ELMO - Brief Overview
# 
# Key idea:
# 
# * Train a `forward LSTM-based language model` and a `backward LSTM-based language model` on some large corpus.
# * Use the hidden states of the LSTMs for each token to compute a vector representation of each word.
# 
# A step in the pre-training process of ELMo: Given `"Let's stick to"` as input, predict the next most likely word – a language modeling task. When trained on a large dataset, the model starts to pick up on language patterns. It’s unlikely it’ll accurately guess the next word in this example. 
# 
# More realistically, after a word such as **hang**, it will assign a higher probability to a word like **out** (to spell `hang out`) than to **camera**.
# 
# > We can see the hidden state of each unrolled-LSTM step peaking out from behind ELMo’s head. Those come in handy in the embedding proecss after this pre-training is done.
# 
# **ELMo** actually goes a step further and trains a **bi-directional LSTM** – so that its language model doesn't only have a sense of the `next word`, but also the `previous word`.
# 
# ![](images/elmo_step1.png)
# 
# **ELMo** comes up with the [contextualized embedding](https://pythonandml.github.io/dlbook/content/word_embeddings/contextual_word_embeddings.html) through grouping together the hidden states (and initial embedding) in a certain way (concatenation followed by weighted summation).
# 
# ![](images/elmo_step2.png)
# 
# [Image source for both the images](https://jalammar.github.io/illustrated-bert/)

# ### Implementation of ELMo word embeddings using Python-Tensorflow

# Run these command before running the code in your terminal to install the necessary libraries.
# 
# ```
# pip install "tensorflow>=2.0.0"
# pip install --upgrade tensorflow-hub
# ```

# #### import necessary libraries
# 

# In[1]:


import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


# #### Load pre trained ELMo model
# 

# In[2]:


elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)


# #### Create an instance of ELMo
# 

# In[3]:


documents = ["I will show you a valid point of reference and talk to the point",
		         "Where have you placed the point"]


# In[5]:


embeddings = elmo(documents,
	                signature="default",
	                as_dict=True)["elmo"]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# #### Print word embeddings for word point in given two sentences

# In[11]:


print("Word embeddings for the first 'point' in first sentence")
print(sess.run(embeddings[0][6]))


# In[12]:


print("Word embeddings for the second 'point' in first sentence")
print(sess.run(embeddings[0][-1]))


# In[13]:


print("Word embeddings for 'point' in second sentence")
print(sess.run(embeddings[1][-1]))


# The output shows different word embeddings for the same word **point** used in a different context in different sentences (also the embedding is different in case of same sentence but different context).
