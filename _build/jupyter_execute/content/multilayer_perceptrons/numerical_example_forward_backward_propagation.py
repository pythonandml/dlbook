#!/usr/bin/env python
# coding: utf-8

# # 2.12. Numerical example Forward and Back pass
# 
# Here we present **Numerical example (with code) - Forward pass and Backpropagation (step by step vectorized form)**
# 
# **Note:** 
# 
# *  The equations (in vectorized form) for forward propagation can be found [here](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/forward_propagation.html) (link to previous chapter) 
# 
# *  The equations (in vectorized form) for back propagation can be found [here](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/backpropagation.html) (link to previous chapter) 
# 
# Consider the network shown

# ![](images/neural_nets_architecture_2.png)

# **Given values**
# 
# Input $x = [1, 4, 5]$, $y = [0.1, 0.05]$
# 
# $$
# W_1 = \begin{bmatrix}
# 0.1 & 0.2\\ 
# 0.3 & 0.4\\ 
# 0.5 & 0.6
# \end{bmatrix}
# $$
# 
# $$
# b_1 = \begin{bmatrix}
# 0.5\\ 
# 0.5
# \end{bmatrix}
# $$
# 
# $$
# W_2 = \begin{bmatrix}
# 0.7 & 0.8\\ 
# 0.9 & 0.1
# \end{bmatrix}
# $$
# 
# $$
# b_2 = \begin{bmatrix}
# 0.5\\ 
# 0.5
# \end{bmatrix}
# $$
# 
# The [activation functions](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/activation.html) $f_1(z)$ and $f_2(z)$ (link to previous chapter) used here is **sigmoid** (for both the layers) and the [cost function](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/cost_functions.html) $J(W, b)$ (link to previous chapter) is **MSE**.
# 
# > **Note:** $\odot$ means element wise multiplication (also called **Hadamard product**)
# 

# Let us write the code simultaneously

# In[ ]:


import numpy as np

def sigmoid(x):
    '''
    Parameters
    
    x: input matrix of shape (m, d) 
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    '''
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    # sigmoid derivative
    return sigmoid(x) * (1-sigmoid(x))

def d_mse(a, y):
    '''
    dJ/daL
    '''
    return a - y

x = np.array([[1, 4, 5]])
y = np.array([[0.1, 0.05]])

W1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])
b1 = np.array([[0.5],
               [0.5]])

W2 = np.array([[0.7, 0.8],
               [0.9, 0.1]])
b2 = np.array([[0.5],
               [0.5]])

print("Forward and Backpropagation - Numerical example")
print("\nx =", x)
print("\ny =", y)
print("\nW1 =\n\n", W1)
print("\nb1 =\n\n", b1)
print("\nW2 =\n\n", W2)
print("\nb2 =\n\n", b2)


# **Fowrard Propagation**
# 
# For input layer ($l=0$),
# 
# $$z_0 = a_0 = x$$
# 
# For hidden layer ($l=1$)
# 
# $$
# z_1 = a_0 W_1 + b_1^T
# $$
# 
# $$
# \therefore z_1 = \begin{bmatrix}
# 1 & 4 & 5
# \end{bmatrix} \begin{bmatrix}
# 0.1 & 0.2\\ 
# 0.3 & 0.4\\ 
# 0.5 & 0.6
# \end{bmatrix} + \begin{bmatrix}
# 0.5 & 0.5
# \end{bmatrix}
# $$
# 
# $$
# \therefore z_1 = \begin{bmatrix}
# 3.8 & 4.8
# \end{bmatrix} + \begin{bmatrix}
# 0.5 & 0.5
# \end{bmatrix}
# $$
# 
# $$
# \therefore z_1 = \begin{bmatrix}
# 4.3 & 5.3
# \end{bmatrix}
# $$
# 
# Now, 
# 
# $$a_1 = f_1(z_1) = f_1(\begin{bmatrix}
# 4.3 & 5.3
# \end{bmatrix})$$
# 
# $$a_1 = \begin{bmatrix}
# 0.9866 & 0.9950
# \end{bmatrix}$$
# 
# 

# Let us code the same

# In[ ]:


z0 = a0 = x.copy()
z1 = a0 @ W1 + b1.T
a1 = sigmoid(z1)
print("z1 =", z1)
print("\na1 =", a1)


# Now for the output layer ($l=2$),
# 
# $$
# z_2 = a_1 W_2 + b_2^T
# $$
# 
# $$
# \therefore z_2 = \begin{bmatrix}
# 0.9866 & 0.9950
# \end{bmatrix} \begin{bmatrix}
# 0.7 & 0.8\\ 
# 0.9 & 0.1
# \end{bmatrix} + \begin{bmatrix}
# 0.5 & 0.5
# \end{bmatrix}
# $$
# 
# $$
# \therefore z_2 = \begin{bmatrix}
# 2.086 & 1.389
# \end{bmatrix}
# $$
# 
# Now, 
# 
# $$a_2 = f_2(z_2) = f_2(\begin{bmatrix}
# 2.086 & 1.389
# \end{bmatrix})$$
# 
# $$a_2 = \begin{bmatrix}
# 0.8896 & 0.8004
# \end{bmatrix}$$
# 
# 

# In[ ]:


z2 = a1 @ W2 + b2.T
a2 = sigmoid(z2)
print("z2 =", z2)
print("\na2 =", a2)


# **Backpropagation**
# 
# Output layer Backpropagation error ($\delta_2$)
# 
# $$
# \frac{\partial J(W, b)}{\partial a_2} = a_2 - y = \begin{bmatrix}
# 0.8896 & 0.8004
# \end{bmatrix} - \begin{bmatrix}
# 0.1 & 0.05
# \end{bmatrix}
# $$
# 
# $$
# \frac{\partial J(W, b)}{\partial a_2} = \begin{bmatrix}
# 0.7896 & 0.7504
# \end{bmatrix}
# $$
# 
# $$
# f'_2(z_2) = f'_2(\begin{bmatrix}
# 2.086 & 1.389
# \end{bmatrix}) = \begin{bmatrix}
# 0.0983 & 0.1598
# \end{bmatrix}
# $$
# 
# Therefore, 
# 
# $$
# \delta_2 = \frac{\partial J(W, b)}{\partial a_2} \odot f'_2(z_2)
# $$
# 
# $$
# \delta_2 = \begin{bmatrix}
# 0.7896 & 0.7504
# \end{bmatrix} \odot \begin{bmatrix}
# 0.0983 & 0.1598
# \end{bmatrix}
# $$
# 
# $$
# \delta_2 = \begin{bmatrix}
# 0.0776 & 0.1199
# \end{bmatrix}
# $$
# 
# 

# In[ ]:


dJda2 = d_mse(a2, y)
da2dz2 = d_sigmoid(z2)
d2 = dJda2 * da2dz2

print("dJda2 =",dJda2)
print("\nf2'(z2) =",da2dz2)
print("\nd2 =",d2)


# Hidden layer Backpropagation error ($\delta_1$)
# 
# $$
# \delta_1 = (\delta_2 W_2^T) \odot f'_1(z_1)
# $$
# 
# $$
# f'_1(z_1) = f'_1(\begin{bmatrix}
# 4.3 & 5.3
# \end{bmatrix}) = \begin{bmatrix}
# 0.0132 & 0.0049
# \end{bmatrix}
# $$
# 
# Therefore,
# 
# $$
# \delta_1 = \begin{bmatrix}
# 0.0776 & 0.1199
# \end{bmatrix} \begin{bmatrix}
# 0.7 & 0.9\\ 
# 0.8 & 0.1
# \end{bmatrix} \odot \begin{bmatrix}
# 0.0132 & 0.0049
# \end{bmatrix}
# $$
# 
# $$
# \delta_1 = \begin{bmatrix}
# 0.002 & 0.0004
# \end{bmatrix}
# $$
# 
# 

# In[ ]:


da1dz1 = d_sigmoid(z1)
d1 = (d2 @ W2.T) * da1dz1

print("f1'(z1) =",da1dz1)
print("\nd1 =",d1)


# Rate of change of the cost with respect to weights $W_l$
# 
# For $l=1$,
# 
# $$
# \frac{\partial J(W, b)}{\partial W_1} = a_0^T \delta_1
# $$
# 
# $$
# \frac{\partial J(W, b)}{\partial W_1} = \begin{bmatrix}
# 1\\ 
# 4\\ 
# 5
# \end{bmatrix} \begin{bmatrix}
# 0.002 & 0.0004
# \end{bmatrix}
# $$
# 
# $$
# \frac{\partial J(W, b)}{\partial W_1} = \begin{bmatrix}
# 0.002 & 0.0004 \\ 
# 0.0079 & 0.0016\\ 
# 0.0099 & 0.002
# \end{bmatrix}
# $$
# 

# In[ ]:


dLdW1 = a0.T @ d1

print('dLdW1 =\n\n', np.round(dLdW1, 4))


# For $l=2$,
# 
# $$
# \frac{\partial J(W, b)}{\partial W_2} = a_1^T \delta_2
# $$
# 
# $$
# \frac{\partial J(W, b)}{\partial W_2} = \begin{bmatrix}
# 0.0765 & 0.1183\\ 
# 0.0772 & 0.1193
# \end{bmatrix}
# $$
# 
# 

# In[ ]:


dLdW2 = a1.T @ d2
print('dLdW2 =\n\n', np.round(dLdW2, 4))


# Rate of change of the cost with respect to bias $b_l$
# 
# Finally, the partial derivative of the cost function $J(W, b)$ with respect to bias of that layer $b_l$ will be:
# 
# For $l=1$,
# 
# $$
# \frac{\partial J(W, b)}{\partial b_1} = \sum \delta_1 = \begin{bmatrix}
# 0.002\\ 
# 0.0004
# \end{bmatrix}
# $$
# 
# 

# In[ ]:


dLdb1 = np.sum(d1, axis=0).reshape(-1,1)
print('dLdb1 =\n\n', np.round(dLdb1, 4))


# For $l=2$,
# 
# $$
# \frac{\partial J(W, b)}{\partial b_2} = \sum \delta_2 = \begin{bmatrix}
# 0.0775\\ 
# 0.1199
# \end{bmatrix}
# $$
# 

# In[ ]:


dLdb2 = np.sum(d2, axis=0).reshape(-1,1)
print('dLdb2 =\n\n', np.round(dLdb2, 4))


# **Update the parameters**
# 
# > **Note:** Although this has not been introduced yet in our chapters, but just for the sake of completenss, we will show how to update the weights and biases using the partial derivatives obtained. So, if you are not aware of this step then you can skip it for now.
# 
# Let the learning rate $\eta = 0.01$.
# 
# **Updating $W_1$**
# 
# $$
# W_1 := W_1 - \frac{\eta}{m} \frac{\partial J(W, b)}{\partial W_1}
# $$
# 
# $$
# \therefore W_1 = \begin{bmatrix}
# 0.1 & 0.2\\ 
# 0.3 & 0.4\\ 
# 0.5 & 0.6
# \end{bmatrix} - 0.01 \begin{bmatrix}
# 0.002 & 0.0004 \\ 
# 0.0079 & 0.0016\\ 
# 0.0099 & 0.002
# \end{bmatrix}
# $$
# 
# $$
# \therefore W_1 = \begin{bmatrix}
# 0.09998 & 0.1999\\ 
# 0.29992 & 0.39998\\ 
# 0.4999 & 0.59998
# \end{bmatrix}
# $$
# 
# **Updating $W_2$**
# 
# $$
# W_2 := W_2 - \frac{\eta}{m} \frac{\partial J(W, b)}{\partial W_2}
# $$
# 
# $$
# \therefore W_2 = \begin{bmatrix}
# 0.7 & 0.8\\ 
# 0.9 & 0.1
# \end{bmatrix} - 0.01 \begin{bmatrix}
# 0.0765 & 0.1183\\ 
# 0.0772 & 0.1193
# \end{bmatrix}
# $$
# 
# $$
# \therefore W_2 = \begin{bmatrix}
# 0.6992 & 0.7988\\ 
# 0.89923 &  0.0988
# \end{bmatrix}
# $$
# 
# **Updating $b_1$**
# 
# $$
# b_1 := b_1 - \frac{\eta}{m} \frac{\partial J(W, b)}{\partial b_1}
# $$
# 
# $$
# \therefore b_1 = \begin{bmatrix}
# 0.5\\ 
# 0.5
# \end{bmatrix} - 0.01 \begin{bmatrix}
# 0.002\\ 
# 0.0004
# \end{bmatrix}
# $$
# 
# $$
# \therefore b_1 = \begin{bmatrix}
# 0.49998\\ 
# 0.499995
# \end{bmatrix}
# $$
# 
# **Updating $b_2$**
# 
# $$
# b_2 := b_2 - \frac{\eta}{m} \frac{\partial J(W, b)}{\partial b_2}
# $$
# 
# $$
# \therefore b_2 = \begin{bmatrix}
# 0.5\\ 
# 0.5
# \end{bmatrix} - 0.01 \begin{bmatrix}
# 0.0775\\ 
# 0.1199
# \end{bmatrix}
# $$
# 
# $$
# \therefore b_2 = \begin{bmatrix}
# 0.49922\\ 
# 0.49880
# \end{bmatrix}
# $$
# 

# In[ ]:


n = 0.01
m = y.shape[0]


# In[ ]:


W1n = W1 - (n/m)*dLdW1
print("Updated Weight W1 =\n\n", W1n)


# In[ ]:


W2n = W2 - (n/m)*dLdW2
print("Updated Weight W2 =\n\n", W2n)


# In[ ]:


b1n = b1 - (n/m)*dLdb1
print("Updated bias b1 =\n\n", b1n)


# In[ ]:


b2n = b2 - (n/m)*dLdb2
print("Updated bias b2 =\n\n", b2n)


# The solution (in non-vectorized format) for the given network can be found [here](https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/) (link to an external website). On comparing we find our solution is in complete agreement with that solution! We can easily extend this vectorized format for multiple hidden layers as well as for a batch dataset. You can also repeat this update for many epochs. Complete code is shown below

# In[ ]:


import numpy as np

# Utility functions for activation, cost and their derivatives

def sigmoid(x):
    '''
    Parameters
    
    x: input matrix of shape (m, d) 
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    '''
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    # sigmoid derivative
    return sigmoid(x) * (1-sigmoid(x))

def d_mse(a, y):
    '''
    dJ/daL
    '''
    return a - y

# Given Parameters

x = np.array([[1, 4, 5]])
y = np.array([[0.1, 0.05]])

W1 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])
b1 = np.array([[0.5],
               [0.5]])

W2 = np.array([[0.7, 0.8],
               [0.9, 0.1]])
b2 = np.array([[0.5],
               [0.5]])

# Forward Propagation

z0 = a0 = x.copy()
z1 = a0 @ W1 + b1.T
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2.T
a2 = sigmoid(z2)

# Backward Propagation

# 1. Output error

dJda2 = d_mse(a2, y)
da2dz2 = d_sigmoid(z2)
d2 = dJda2 * da2dz2

# 2. Hidden layer error

da1dz1 = d_sigmoid(z1)
d1 = (d2 @ W2.T) * da1dz1

# 3. dJ/dW

dLdW1 = a0.T @ d1
dLdW2 = a1.T @ d2

# 4. dJ/db

dLdb1 = np.sum(d1, axis=0).reshape(-1,1)
dLdb2 = np.sum(d2, axis=0).reshape(-1,1)

# Update parameters

n = 0.01 # Learning rate
m = y.shape[0]

W1n = W1 - (n/m)*dLdW1
W2n = W2 - (n/m)*dLdW2
b1n = b1 - (n/m)*dLdb1
b2n = b2 - (n/m)*dLdb2

# Prints

print("Forward and Backpropagation - Numerical example")
print("\nx =", x)
print("\ny =", y)
print("\nW1 =\n\n", W1)
print("\nb1 =\n\n", b1)
print("\nW2 =\n\n", W2)
print("\nb2 =\n\n", b2)
print("\nz1 =", z1)
print("\na1 =", a1)
print("\nz2 =", z2)
print("\na2 =", a2)
print("\ndJda2 =",dJda2)
print("\nf2'(z2) =",da2dz2)
print("\nd2 =",d2)
print("\nf1'(z1) =",da1dz1)
print("\nd1 =",d1)
print('\ndLdW1 =\n\n', np.round(dLdW1, 4))
print('\ndLdW2 =\n\n', np.round(dLdW2, 4))
print('\ndLdb1 =\n\n', np.round(dLdb1, 4))
print('\ndLdb2 =\n\n', np.round(dLdb2, 4))
print("\nUpdated Weight W1 =\n\n", W1n)
print("\nUpdated Weight W2 =\n\n", W2n)
print("\nUpdated bias b1 =\n\n", b1n)
print("\nUpdated bias b2 =\n\n", b2n)

