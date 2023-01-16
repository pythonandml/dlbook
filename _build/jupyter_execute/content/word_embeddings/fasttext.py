#!/usr/bin/env python
# coding: utf-8

# # 4.2.3. FastText
# 
# A common problem in Natural Processing Language (NLP) tasks is to capture the context in which the word has been used. A single word with the same spelling and pronunciation (`homonyms`) can be used in multiple contexts and a potential solution to the above problem is making word embeddings.
# 
# **FastText** is a library created by the Facebook Research Team for efficient learning of word representations like [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html) (link to previous chapter) or [GloVe](https://pythonandml.github.io/dlbook/content/word_embeddings/glove.html) (link to previous chapter) and sentence classification and is a type of [static word embedding](https://pythonandml.github.io/dlbook/content/word_embeddings/static_word_embeddings.html) (link to previous chapter). If you want you can read the official [fastText paper](https://arxiv.org/pdf/1607.04606.pdf).
# 
# :::{note}
# `FastText` differs in the sense that [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html) (link to previous chapter) treats every **single word** as the **smallest unit** whose vector representation is to be found but `FastText` assumes a word to be formed by a **n-grams of character**. 
# 
# For example:
# 
# word `sunny` is composed of `[sun, sunn, sunny], [sunny, unny, nny]`  etc, where $n$ could range from 1 to the length of the word.
# :::
# 
# **Examples of different length character n-grams are given below:**
# 
# ![](images/fasttext_3_grams_list.png)
# 
# [Image Source](https://amitness.com/2020/06/fasttext-embeddings/)
# 
# <table>
# <thead>
# <tr>
# <th>Word</th>
# <th>Length(n)</th>
# <th>Character n-grams</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# <td>eating</td>
# <td>3</td>
# <td>&lt;ea, eat, ati, tin, ing, ng&gt;</td>
# </tr>
# <tr>
# <td>eating</td>
# <td>4</td>
# <td>&lt;eat, eati, atin, ting, ing&gt;</td>
# </tr>
# <tr>
# <td>eating</td>
# <td>5</td>
# <td>&lt;eati, eatin, ating, ting&gt;</td>
# </tr>
# <tr>
# <td>eating</td>
# <td>6</td>
# <td>&lt;eatin, eating, ating&gt;</td>
# </tr>
# </tbody>
# </table>
# 
# Thus **FastText** works well with rare words. So, even if a word wasn't seen during training, it can be broken down into n-grams to get its embeddings. [Word2Vec](https://pythonandml.github.io/dlbook/content/word_embeddings/word2vec.html) (link to previous chapter) and [GloVe](https://pythonandml.github.io/dlbook/content/word_embeddings/glove.html) (link to previous chapter) both fail to provide any vector representation for words that are not in the model dictionary. This is a huge advantage of this method.
# 
# ### FastText model from python genism library
# 
# To train your own embeddings, you can either use the [official CLI tool](https://fasttext.cc/docs/en/unsupervised-tutorial.html) or use the fasttext implementation available in gensim.
# 
# You can install and import gensim library and then use gensim library to extract most similar words from the model that you downloaded from FastText.
# 
# Assume we use the same corpus as we have used in the [GloVe](https://pythonandml.github.io/dlbook/content/word_embeddings/glove.html) (link to previous chapter) model
# 
# #### Import essential libraries

# In[1]:


from gensim.models.fasttext import FastText


# In[3]:


documents = ['this is the first document',
             'this document is the second document',
             'this is the third one',
             'is this the first document']


# **Tokenize**
# 
# We will first tokenize the above documents list (extract words from the sentences).
# 

# In[4]:


word_tokens = []

for document in documents:
    words = []
    for word in document.split(" "):
        words.append(word)
    word_tokens.append(words)

word_tokens


# #### Defining values for parameters
# 
# The hyperparameters used in this model are:
# 
# * `size`: Dimensionality of the word vectors. window=window_size,
# * `min_count`: The model ignores all words with total frequency lower than this.
# * `sample`: The threshold for configuring which higher-frequency words are randomly down sampled, useful range is (0, 1e-5).
# * `workers`: Use these many worker threads to train the model (=faster training with multicore machines).
# * `sg`: Training algorithm: skip-gram if sg=1, otherwise CBOW.
# * `iter`: Number of iterations (epochs) over the corpus.

# In[9]:


embedding_size = 300
window_size = 2
min_word = 1
down_sampling = 1e-2


# Let’s train Gensim fastText word embeddings model with our own custom data:

# In[10]:


fast_Text_model = FastText(word_tokens,
                      size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      workers = 4,
                      sg=1,
                      iter=100)


# #### Explore Gensim fastText model

# In[13]:


# Check word embedding for a perticular word

fast_Text_model.wv['document'].shape


# In[14]:


# Check top 5 similar word for a given word by gensim fastText

fast_Text_model.wv.most_similar('first', topn=5)


# In[16]:


# Check similarity score between two word

fast_Text_model.wv.similarity('second', 'first')


# ### FastText models from Official CLI tool
# 
# #### Building fasttext python module
# 
# In order to build fasttext module for python, use the following:
# 

# In[19]:


get_ipython().system('git clone https://github.com/facebookresearch/fastText.git')


# In[ ]:


get_ipython().run_line_magic('cd', 'fastText')
get_ipython().system('make')
get_ipython().system('cp fasttext ../')
get_ipython().run_line_magic('cd', '..')


# In[27]:


get_ipython().system('./fasttext')


# If everything was installed correctly then, you should see the list of available commands for FastText as the output.
# 
# If you want to learn word representations using **Skipgram** and **CBOW models** from FastText model, we will see how we can implement both these methods to learn vector representations for a sample text file [file.txt](https://github.com/pythonandml/dlbook/blob/main/content/word_embeddings/datasets/file.txt) using fasttext.
# 
# **Skipgram**
# 
# > ./fasttext skipgram -input file.txt -output model
# 
# **CBOW**
# 
# > ./fasttext cbow -input file.txt -output model
# 
# Let us see the parameters defined above in steps for easy understanding.
# 
# `./fasttext` – It is used to invoke the FastText library.
# 
# `skipgram/cbow` – It is where you specify whether skipgram or cbow is to be used to create the word representations.
# 
# `-input` – This is the name of the parameter which specifies the following word to be used as the name of the file used for training. This argument should be used as is.
# 
# `data.txt` – a sample text file over which we wish to train the skipgram or cbow model. Change this name to the name of the text file you have.
# 
# `-output` – This is the name of the parameter which specifies the following word to be used as the name of the model being created. This argument is to be used as is.
# 
# `model` – This is the name of the model created.
# 
# Running the above command will create two files named `model.bin` and `model.vec`. 
# 
# **model.bin** contains the model parameters, dictionary and the hyperparameters and can be used to compute word vectors. 
# 
# **model.vec** is a text file that contains the word vectors for one word per line.

# In[33]:


get_ipython().system('./fasttext skipgram -input file.txt -output model')


# #### Print word vectors of a word
# 
# In order to get the word vectors for a word or set of words, save them in a text file. For example, here is a sample text file named [queries.txt](https://github.com/pythonandml/dlbook/blob/main/content/word_embeddings/datasets/queries.txt) that contains some random words. 
# 
# > This is a sample document whose word vectors I want to calculate per line.
# 
# We will get the vector representation of these words using the model we trained above.

# In[34]:


get_ipython().system('./fasttext print-word-vectors model.bin < queries.txt')


# To check word vectors for a single word without saving into a file, you can do:
# 
# 

# In[35]:


get_ipython().system('echo "word" | ./fasttext print-word-vectors model.bin')


# #### Finding similar words
# 
# You can also find the words most similar to a given word. This functionality is provided by the nn parameter. Let’s see how we can find the most similar words to “happy”.

# In[36]:


get_ipython().system('./fasttext nn model.bin')


# **Explore further**
# 
# [Code Source for Official CLI tool section](https://www.analyticsvidhya.com/blog/2017/07/word-representations-text-classification-using-fasttext-nlp-facebook/)
# 
# [Code Source for gensim part](https://thinkinfi.com/fasttext-word-embeddings-python-implementation/)
