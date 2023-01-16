#!/usr/bin/env python
# coding: utf-8

# # 2.16. MLP model using Tensorflow - Keras
# 
# After [Building Neural Network (Multi Layer Perceptron model) from scratch using Numpy in Python](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/neural_networks_mlp_scratch_best.html) (link to previous chapter), and after developing [MLP using Pytorch](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/mlp_pytorch.html) (link to previous chapter), we will finally develop the MLP model using Tensorflow - Keras.
# 
# > **Note:** The MLP model we developed from scratch almost follows the way the models are developed in Keras. 

# #### Import necessary libraries
# 
# Here we import a `Dense` layer, an `Activation` layer and a `Dropout` layer. Then we will also import optimizers `Adam` and `RMSprop`.
# 
# Then we finally import the `to_categorical` function which is nothing but one hot vector function.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt # plotting library
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
from keras.layers import Dense , Activation, Dropout
from keras.optimizers import Adam ,RMSprop
from keras.utils import to_categorical


# #### Data Loading and pre-processing

# Next we import and load the **MNIST** dataset
# 
# MNIST is a collection of handwritten digits ranging from the number 0 to 9.
# 
# It has a training set of 60,000 images, and 10,000 test images that are classified into corresponding categories or labels.

# In[ ]:


# import dataset
from keras.datasets import mnist

# load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# After loading the MNIST dataset, the number of labels is computed as:

# In[ ]:


# compute the number of labels
num_labels = len(np.unique(y_train))


# Now we will perform [One hot vector encoding](https://pythonandml.github.io/dlbook/content/preliminaries/data_preprocessing.html#one-hot-encoding) (link to previous chapter) on the target data

# In[ ]:


# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# Let us define our `input_shape`

# In[ ]:


input_size = x_train.shape[1] * x_train.shape[1]
input_size


# Now we will **resize** and **normalize** the data

# In[ ]:


# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255


# Now, we will set the network parameters as follows:
# 

# In[ ]:


# network parameters
batch_size = 128
hidden_units = 256
dropout = 0.45


# #### Model architecture
# 
# The next step is to design the model architecture. The proposed model is made of three MLP layers.
# 
# In Keras, a Dense layer stands for the densely (fully) connected layer.
# 
# Our model is a **3-layer MLP** with *ReLU and dropout* after each layer

# In[ ]:


model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))


# Keras library provides us summary() method to check the model description.

# In[ ]:


model.summary()


# #### Executing the MLP model using Keras 
# 
# This section comprises of 
# 
# * Compiling the model with the compile() method.
# 
# * Training the model with fit() method.
# 
# * Evaluating the model performance with evaluate() method.

# Compiling the model

# In[ ]:


model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])


# Training the model

# In[ ]:


model.fit(x_train, y_train, epochs=20, batch_size=batch_size)


# Evaluating model performance with evaluate() method 

# In[ ]:


loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))


# We get the test accuracy of 98.2%. It is that simple!
