#!/usr/bin/env python
# coding: utf-8

# # 4.3. Contextual Word Embeddings
# 
# In order to understand what word embeddings are, why do we need it and what are traditional word embeddings, please visit [this page](https://pythonandml.github.io/dlbook/content/word_embeddings/traditional_word_embeddings.html) (link to the previous chapter). 
# 
# As discussed in [static word embedding](https://pythonandml.github.io/dlbook/content/word_embeddings/static_word_embeddings.html) (link to previous chapter), 
# 
# Consider the two sentences:
# 
# 1. I will show you a valid **point** of reference and talk to the **point**.
# 
# 2. Where have you placed the **point**.
# 
# > Now, for the [static word embeddings](https://pythonandml.github.io/dlbook/content/word_embeddings/static_word_embeddings.html) (link to previous chapter) from a pre-trained embeddings such as [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html) (link to previous chapter), the embeddings for the word **point** is same for both of its occurrences in example-1 and also the same for the word **point** in example-2. All three occurrences has same embeddings.
# 
# These are known as **Polysemous words** where the word use (e.g., syntax and
# semantics) depends on its `context`. So, why not learn the
# representations for each word in its context?
# 
# ![](images/point_contextual.png)
# 
# Therefore the concept of `Contextualized Word Embeddings` came into the picture, which returns different embeddings for the same word depending on the words around it — its embeddings are **context-sensitive**. 
# 
# It would actually return different answers for **point** in both of these examples because it would recognize that the word is being used in different contexts. Let us explore further different types of **contextualized word embeddings**.
# 
# 1. [ELMo](https://pythonandml.github.io/dlbook/content/word_embeddings/elmo.html)
# 
# 2. [BERT](https://pythonandml.github.io/dlbook/content/word_embeddings/bert.html)
# 
# 
# 
