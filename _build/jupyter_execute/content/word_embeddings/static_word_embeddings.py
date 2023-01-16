#!/usr/bin/env python
# coding: utf-8

# # 4.2. Static Word Embeddings
# 
# In order to understand what word embeddings are, why do we need it and what are traditional word embeddings, please visit [this page](https://pythonandml.github.io/dlbook/content/word_embeddings/traditional_word_embeddings.html) (link to the previous chapter).
# 
# In this section we will be describing various ways to find the vector representation of a word using the `Static Word Embeddings` approach.
# 
# `Static Word embeddings` and [Contextual Word embeddings](https://pythonandml.github.io/dlbook/content/word_embeddings/contextual_word_embeddings.html) are slightly different.
# 
# Word embeddings provided by **word2vec**, **Glove** or **fastText** has a vocabulary (dictionary) of words. The elements of this vocabulary (or dictionary) are words and its corresponding word embeddings. Hence, given a word, its embeddings is always the same in whichever sentence it occurs. Here, the pre-trained word embeddings are static. 
# 
# For example, consider the two sentences:
# 
# 1. I will show you a valid **point** of reference and talk to the **point**.
# 
# 2. Where have you placed the **point**.
# 
# > Now, for the static word embeddings from a pre-trained embeddings such as [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html) (link to previous chapter), the embeddings for the word **point** is same for both of its occurrences in example-1 and also the same for the word **point** in example-2. All three occurrences has same embeddings ([reference](https://stackoverflow.com/a/62314668/20878502)).
# 
# However note that the context of a `single contextual word` is mostly preserved in this type of embedding. Let us explore further different types of static word embeddings.
# 
# 1. [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html)
# 
# 2. [GloVe](https://pythonandml.github.io/dlbook/content/word_embeddings/glove.html)
# 
# 3. [FastText](https://pythonandml.github.io/dlbook/content/word_embeddings/fasttext.html)
# 
# 

# ### Cosine Similarity
# 
# It is the most widely used method to compare two vectors. It is a dot product between two vectors. We would find the cosine angle between the two vectors. For degree 0, cosine is 1 and it is less than 1 for any other angle.
# 
# ![](images/similarity_cosine.webp)
# 
# Let us compute cosine similarity between 2 vectors using sklearn's cosine similarity module.

# In[2]:


from sklearn.metrics.pairwise import cosine_similarity

A = [[1, 3]]
B = [[-2, 2]]

print("Cosine Similarity between A and B =", cosine_similarity(A, B))

