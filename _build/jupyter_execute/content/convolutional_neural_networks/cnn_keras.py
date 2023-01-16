#!/usr/bin/env python
# coding: utf-8

# # 3.5. CNN model using Tensorflow - Keras
# 
# After [Building Convolutional Neural Network (CNN model) from scratch using Numpy in Python](https://pythonandml.github.io/dlbook/content/convolutional_neural_networks/cnn_from_scratch.html) (link to previous chapter), and after developing [CNN using Pytorch](https://pythonandml.github.io/dlbook/content/convolutional_neural_networks/cnn_pytorch.html) (link to previous chapter), we will finally develop the CNN model using Tensorflow - Keras.
# 
# > **Note:** The CNN model we developed from scratch almost follows the way the models are developed in Keras. 

# #### Import necessary libraries
# 
# Here we import a `Conv2D`, `MaxPooling2D`, `Dense` layer, an `Activation` layer and a `Dropout` layer. Then we will also import optimizers `Adam` and `RMSprop`.
# 
# Then we finally import the `to_categorical` function which is nothing but one hot vector function.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt # plotting library
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam ,RMSprop
from keras.utils import to_categorical


# #### Data Loading and pre-processing

# Next we import and load the **CIFAR-10** dataset
# 
# ```{note}
# CIFAR-10 is a dataset that has a collection of images of 10 different classes. This dataset is widely used for research purposes to test different machine learning models and especially for computer vision problems.
# ```

# In[2]:


# import dataset
from keras.datasets import cifar10

# load dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


# After loading the CIFAR-10 dataset, the number of labels is computed as:

# In[3]:


# compute the number of labels
num_labels = len(np.unique(y_train))


# Now we will perform [One hot vector encoding](https://pythonandml.github.io/dlbook/content/preliminaries/data_preprocessing.html#one-hot-encoding) (link to previous chapter) on the target data

# In[4]:


# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Now we will **normalize** the data

# In[5]:


# normalize
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


# #### Model architecture
# 
# The next step is to design the model architecture.

# In[7]:


# Creating a sequential model and adding layers to it

model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax')) # number of classes (output) = 10


# Keras library provides us summary() method to check the model description.

# In[8]:


model.summary()


# #### Executing the CNN model using Keras 
# 
# This section comprises of 
# 
# * Compiling the model with the compile() method.
# 
# * Training the model with fit() method.
# 
# * Evaluating the model performance with evaluate() method.

# Compiling the model

# In[9]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# Training the model

# In[10]:


model.fit(x_train, y_train, epochs=5, batch_size=64)


# Evaluating model performance with evaluate() method 

# In[11]:


loss, acc = model.evaluate(x_test, y_test, batch_size=64)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


# We get the test accuracy of 71.1%. With more complex model, we can increase the accuracy of CIFAR-10 as much as we want. The main thing is that we have learnt how to build our very first CNN model using `Keras`. It is that simple! 
