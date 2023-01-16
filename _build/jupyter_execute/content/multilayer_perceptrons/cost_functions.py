#!/usr/bin/env python
# coding: utf-8

# # 2.4. Cost functions
# 
# The goal of a Neural network model (or any model) is to make as accurate predictions ($\hat{y}$) as possible. For this, we'd like an algorithm which lets us find the parameters (**weights** and **biases**) of the model so that the output from the network ($\hat{y}$) approximates $y$ for the training input $X$. 
# 
# To quantify how well we're achieving this goal we define a *cost function* $J(W,b)$ (which is also called as *Loss* or *Objective* function) which is a function of the parameters weights $W$ and biases $b$. In simple terms, a cost function is a measure of error between what value your model predicts ($\hat{y}$) and what the value actually is ($y$) and our goal is to minimize this error.
# 
# We will only discuss about 2 types of cost functions (one for regression and one for classification).
# 
# **Cost function for regression**
# 
# In case of regression problem, we define cost function to be similar to the **Mean Square Error (MSE)**.
# 
# Mathematically it is expressed as,
# 
# $$
# J(W,b) = \frac{1}{2} \sum_{i=1}^m ||\hat{y}_i-y_i||^2
# $$
# 
# where $||z||$ is the $L2$ norm of $z$. The partial derivative of $J(W,b)$ with respect to $\hat{y}$:
# 
# $$
# \frac{\partial J(W,b)}{\partial \hat{y}} = \hat{y} - y
# $$

# **Example**
# 
# Suppose,
# 
# $$
# a = \begin{bmatrix}
# 1 & 2\\ 
# 2 & 0\\ 
# 3 & 1
# \end{bmatrix}
# $$
# 
# and
# 
# $$
# y = \begin{bmatrix}
# 1 & 1\\ 
# 1 & 2\\ 
# 4 & 3
# \end{bmatrix}
# $$
# 
# So, for $i=1, 2, 3$
# 
# $$
# ||a_i-y_i|| = \begin{bmatrix}
# \left \| [0, 1] \right \|\\ 
# \left \| [1, -2] \right \|\\ 
# \left \| [-1, -2] \right \|
# \end{bmatrix} = \begin{bmatrix}
# 1\\ 
# \sqrt{5}\\ 
# \sqrt{5} 
# \end{bmatrix}
# $$
# 
# Then,
# 
# $$
# ||a_i-y_i||^2 = \begin{bmatrix}
# 1\\ 
# 5\\ 
# 5 
# \end{bmatrix}
# $$
# 
# $$
# J(W,b) = \frac{1}{2}\sum_{i=1}^3||a_i-y_i||^2 = \frac{1}{2}(5+5+1) = 5.5 
# $$

# The *mse(a,y)* function implemented below computes the cost function for regression where $a$ is the predicted output (same as $\hat{y}$).

# In[ ]:


def mse(a, y):
    '''
    Parameters
    
    a: Predicted output array of shape (m, c)
    y: Actual output array of shape (m, c)
    '''
    return (1/2)*np.sum((np.linalg.norm(a-y, axis=1))**2)


# Let us test this function on the dataset from the above example. Complete solution is listed below

# In[ ]:


import numpy as np

def mse(a, y):
    '''
    Parameters
    
    a: Predicted output array of shape (m, c)
    y: Actual output array of shape (m, c)
    '''
    return (1/2)*np.sum((np.linalg.norm(a-y, axis=1))**2)

a = np.array([[1,2],
              [2,0],
              [3,1]])

y = np.array([[1,1],
             [1,2],
             [4,3]])

cost = mse(a, y)

print("Cost = J(W, b) =", round(cost, 3))


# #### Validating using sklearn's MSE function
# 
# > **Note:** Sklearn's mean square implementation is slightly different. It divides the cost function $J(W,b)$ by the number of samples $m$ as well.
# 
# $$
# J(W,b) = \frac{1}{2m} \sum_{i=1}^m ||\hat{y}_i-y_i||^2
# $$

# In[ ]:


from sklearn.metrics import mean_squared_error
import numpy as np

def mse(a, y):
    '''
    Parameters
    
    a: Predicted output array of shape (m, c)
    y: Actual output array of shape (m, c)
    '''
    return (1/2)*np.sum((np.linalg.norm(a-y, axis=1))**2)

a = np.array([[1,2],
              [2,0],
              [3,1]])

y = np.array([[1,1],
             [1,2],
             [4,3]])

m = y.shape[0]
cost = mse(a, y)

print("Cost (our function) = J(W, b) =", round(cost, 3))
print("Cost (sklearn) = J(W, b) =", mean_squared_error(a, y)*m)


# *d_mse(a, y)* function implemented below computes the partial derivative of the cost function with $\hat{y} = a$

# In[ ]:


def d_mse(a, y):
    return a - y


# **Cost function for classification**
# 
# Categorical Cross-entropy cost (also called as log loss) can be used as a loss function when optimizing classification models like neural networks.
# 
# Let $c$ be the number of classes in the target variable $y$ and $y$ is one-hot encoded. Let $a$ contains the Softmax probability (i.e. the final output obtained after passing it to softmax activation) in the output layer. 
# 
# Mathematically it is expressed as:
# 
# $$
# J(W,b) = -\sum_{i=1}^m \sum_{j=1}^c y_j \odot log(a_j)
# $$
# 
# The partial derivative of $J(W,b)$ with respect to $\hat{y} = a$ is ([check this link](https://stats.stackexchange.com/questions/277203/differentiation-of-cross-entropy)):
# 
# $$
# \frac{\partial J(W,b)}{\partial \hat{y}} = \frac{\partial J(W,b)}{\partial a} = -\frac{y}{a}
# $$
# 
# **Softmax with Categorical cross entropy loss**
# 
# It is interesting to note that in the case of softmax activation function with categorical cross entropy loss, if $a = \text{softmax}(z)$, then the partial derivative of $J(W,b)$ with respect to $z$ is:
# 
# $$
# \frac{\partial J(W,b)}{\partial z} = a - y
# $$

# Using the below example (in which we have $m=3$ samples and $c=4$ classes - target variable) we are testing the *cross_entropy(a, y)* function that we have implemented and validating the same using sklearn's log_loss module.
# 
# > **Note:** Sklearn's log loss (cross entropy) implementation is slightly different. It divides the cost function $J(W,b)$ by the number of samples $m$ as well.
# 
# $$
# J(W,b) = -\frac{1}{m}\sum_{i=1}^m \sum_{j=1}^c y_j \odot log(a_j)
# $$

# In[ ]:


import numpy as np
from sklearn.metrics import log_loss

def cross_entropy(a, y, epsilon=1e-12):
    '''
    Parameters
    
    a: Predicted output array of shape (m, c)
    y: Actual output array of shape (m, c)
    '''
    a = np.clip(a, epsilon, 1. - epsilon)
    return -np.sum(y*np.log(a))

a = np.array([[0.25,0.2,0.3,0.25],
              [0.2,0.4,0.1,0.3],
              [0.1,0.1,0.2,0.6]])

y = np.array([[1,0,0,0],
              [0,0,0,1],
              [0,0,1,0]])

m = y.shape[0]

print("Cross Entropy cost (our function) =", cross_entropy(a, y))
print("Cross Entropy cost (sklearn) =", log_loss(y, a)*m)


# *d_cross_entropy(a, y)* function implemented below computes the partial derivative of the cost function with $\hat{y} = a$

# In[ ]:


def d_cross_entropy(a, y, epsilon=1e-12):
    a = np.clip(a, epsilon, 1. - epsilon)
    return -y/a


# Let us club all this together in a python class *Cost* whose constructor **init** has only one parameter (*cost_type*, which can either be 'mse' or 'cross-entropy', default value of 'mse')
# 
# > **Note:** This class has a getter method for both (cost function and its partial derivative)

# In[ ]:


class Cost:

    def __init__(self, cost_type='mse'):
        '''
        Parameters
        
        cost_type: type of cost function
        available options are 'mse', and 'cross-entropy'
        '''
        self.cost_type = cost_type

    def mse(self, a, y):
        '''
        Parameters
        
        a: Predicted output array of shape (m, d)
        y: Actual output array of shape (m, d)
        '''
        return (1/2)*np.sum((np.linalg.norm(a-y, axis=1))**2)

    def d_mse(self, a, y):
        '''
        dJ/da
        '''
        return a - y

    def cross_entropy(self, a, y, epsilon=1e-12):
        a = np.clip(a, epsilon, 1. - epsilon)
        return -np.sum(y*np.log(a))

    def d_cross_entropy(self, a, y, epsilon=1e-12):
        a = np.clip(a, epsilon, 1. - epsilon)
        return -y/a

    def get_cost(self, a, y):
        if self.cost_type == 'mse':
            return self.mse(a, y)
        elif self.cost_type == 'cross-entropy':
            return self.cross_entropy(a, y)
        else:
            raise ValueError("Valid cost functions are only 'mse', and 'cross-entropy'")

    def get_d_cost(self, a, y):
        if self.cost_type == 'mse':
            return self.d_mse(a, y)
        elif self.cost_type == 'cross-entropy':
            return self.d_cross_entropy(a, y)
        else:
            raise ValueError("Valid cost functions are only 'mse', and 'cross-entropy'")


# Let us test this class

# In[ ]:


import numpy as np
from sklearn.metrics import log_loss

a = np.array([[0.25,0.2,0.3,0.25],
              [0.2,0.4,0.1,0.3],
              [0.1,0.1,0.2,0.6]])

y = np.array([[1,0,0,0],
              [0,0,0,1],
              [0,0,1,0]])

m = y.shape[0]

cost = Cost(cost_type='cross-entropy')
cost_value = cost.get_cost(a, y)
dcost = cost.get_d_cost(a, y)

print("Cross Entropy cost =", cost_value)
print("Cross Entropy cost (sklearn) =", log_loss(y, a)*m)
print("\nCost derivative =\n\n", dcost)

