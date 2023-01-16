#!/usr/bin/env python
# coding: utf-8

# # 2.11. Batch Normalization
# 
# 
# > In neural networks, the output of the first layer feeds into the second layer, the output of the second layer feeds into the third, and so on. When the parameters of a layer change, so does the distribution of inputs to subsequent layers.
# 
# These shifts in input distributions are called as **Internal covariate shift** and they can be problematic for neural networks, especially deep neural networks that could have a large number of layers.
# 
# Batch normalization (BN) is a method intended to mitigate internal covariate shift for neural networks.
# 
# Machine learning methods tend to work better when their input data consists of uncorrelated features with zero mean and unit variance. When training a neural network, we can preprocess the data before feeding it to the network to explicitly decorrelate its features; this will ensure that the first layer of the network sees data that follows a nice distribution. 
# 
# However even if we preprocess the input data, the activations at deeper layers of the network will likely no longer be decorrelated and will no longer have zero mean or unit variance since they are output from earlier layers in the network. Even worse, during the training process the distribution of features at each layer of the network will shift as the weights of each layer are updated.
# 
# To overcome this, at training time, a batch normalization layer normalises all the input features to a unit normal distribution $\mathcal{N}(\mu=0,\sigma=1)$. A running average of the means and standard deviations is kept during training, and at test time these running averages are used to center and normalize features.
# 
# * Adding BN layers leads to faster and better convergence (where better means higher accuracy)
# 
# * Adding BN layers allows us to use higher learning rate ($\eta$) without compromising convergence
# 
# **Implementation**
# 
# In practice, we consider the batch normalization as a standard layer, such as a perceptron, a convolutional layer, an activation function or a dropout layer and it is generally applied after calculating the weighted sum $z_l$ and before applying the non-linear activation function $f_l(z_l)$.
# 
# For any layer $l$, Consider $z$ of size $(m,h_l)$ (where $h_l$ is the number of neurons in that hidden layer) be an input to batch normalization ($\text{BN}$). In this case the batch normalization is defined as follows:
# 
# $$
# \text{BN}_{(\gamma, \beta)}(z) = \gamma \odot \frac{z-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
# $$
# 
# where $\mu$ of size $(h_l,1)$ and $\sigma$ of size $(h_l,1)$ are the respective population mean and standard deviation of $z$ over the full batch (of batch size $m$). Note that we add a small constant $\epsilon > 0$ to the variance estimate to ensure that we never attempt division by zero.
# 
# > After applying standardization, the resulting minibatch has zero mean and unit variance. The variables $\gamma$ of size $(h_l,1)$ and $\beta$ of size $(h_l,1)$ are learned parameters that allow a standardized variable to have any mean and standard deviation.
# 
# In simple terms, zero mean and unit standard deviation can reduce the expressive power of the neural network. To maintain the expressive power of the network, it is common to replace the standardized variable $\hat{z}$ with $\gamma \hat{z} + \beta$ where parameters like $W$ and $b$, $\gamma$ and $\beta$ can also be learned. 
# 
# #### Forward pass and Back Propagation in Batch Normalization Layer
# 
# Let us apply batch normalization ($\text{BN}$) on layer $l$ after the weighted sum and before the activation function.
# 
# **Forward pass Batch Normalization (vectorized)**
# 
# We know from the standard [forward propagation](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/forward_propagation.html) (link to previous chapter) that 
# 
# $$
# z_l = a_{l-1}W_l + b_l^T 
# $$
# 
# This $z_l$ will be an input to batch normalization ($\text{BN}$) and let the output we get from this be $q_l$. Also, let
# 
# $$
# \bar{z_l} = z_l-\mu
# $$
# 
# $$
# \sigma_{inv} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
# $$
# 
# and
# 
# $$
# \hat{z_l} = \bar{z_l} \odot \sigma_{inv}
# $$
# 
# Therefore,
# 
# $$
# q_l = \gamma \odot \hat{z_l} + \beta
# $$
# 
# where the parameters are as defined above. And finally, passing $q_l$ through activation function $f_l(x)$. Fianlly,
# 
# $$
# a_l = f_l(q_l)
# $$
# 
# **Backpropagation Batch Normalization (vectorized)**
# 
# We know from the standard [backward propagation](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/backpropagation.html) (link to previous chapter) that (let us denote the cost function $J(W, b, \gamma, \beta)$ as $J$ for simplicity)
# 
# > **Note:** Let $\sum_c$ denote the sum along the column (i.e. sum of column-1 then sum of column-2 and so on) to get a vector of size $(h_l, 1)$
# 
# $$
# \frac{\partial J}{\partial q_l} = \frac{\partial J}{\partial a_l} \odot \frac{\partial a_l}{\partial q_l} = 
# $$
# 
# $$
# \frac{\partial J}{\partial q_l} = (\delta_{l+1} W_{l+1}^T) \odot f'_l(z_l) 
# $$
# 
# This will serve as an input in calculating the partial derivative of cost function $J$ with respect to $\gamma$, $\beta$ and $z_l$ and its size will be $(m,h_l)$
# 
# **Partial derivative of $J$ with respect to $\beta$**
# 
# $$
# \boxed{\frac{\partial J}{\partial \beta} = \sum_c \frac{\partial J}{\partial q_l}}
# $$
# 
# **Partial derivative of $J$ with respect to $\gamma$**
# 
# $$
# \boxed{\frac{\partial J}{\partial \gamma} = \sum_c \frac{\partial J}{\partial q_l} \odot \hat{z_l}}
# $$
# 
# **Partial derivative of $J$ with respect to $\hat{z_l}$**
# 
# $$
# \frac{\partial J}{\partial \hat{z_l}} = \frac{\partial J}{\partial q_l} \odot \gamma
# $$
# 
# **Partial derivative of $J$ with respect to $\mu$**
# 
# $$
# \frac{\partial J}{\partial \mu} = -\sum_c \frac{\partial J}{\partial q_l} \odot \sigma_{inv}
# $$
# 
# **Partial derivative of $J$ with respect to $\sigma^2$**
# 
# $$
# \frac{\partial J}{\partial \sigma^2} = -\frac{1}{2}\sum_c \frac{\partial J}{\partial q_l} \odot \bar{z_l} \odot \sigma_{inv}^3
# $$
# 
# **Partial derivative of $J$ with respect to $z_l$**
# 
# $$
# \boxed{\frac{\partial J}{\partial z_l} = \frac{\partial J}{\partial \hat{z_l}} \odot \sigma_{inv} + \left ( \frac{2}{m} \right ) \frac{\partial J}{\partial \sigma^2} \odot \bar{z_l} + \left ( \frac{1}{m} \right ) \frac{\partial J}{\partial \mu}}
# $$
# 
# And finally,
# 
# $$
# \delta_l = \frac{\partial J}{\partial z_l}
# $$
# 
# Follow [[1]](https://www.adityaagrawal.net/blog/deep_learning/bprop_batch_norm) or [[2]](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html) (links to external websites) derivations in case you are more interested.

# #### Python code for forward and backward pass of Batch normalization

# This is our input to BN layer ($z_l$)
# 
# `z` represents: $z_l$

# In[ ]:


import numpy as np

np.random.seed(42)
z = np.random.randint(low=0,high=10,size=(7,3))
m, d = z.shape
z


# We next need some initial value of $\gamma$ and $\beta$
# 
# `gamma` represents: $\gamma$
# 
# `beta` represents: $\beta$

# In[ ]:


gamma = np.ones((d))
np.random.seed(24)
beta = np.zeros((d))


# In[ ]:


gamma


# In[ ]:


beta


# **Forward pass**

# `eps` represents: $\epsilon$
# 
# `mu` represents: $\mu$
# 
# `var` represents: $\sigma^2$
# 
# `zmu` represents: $\bar{z_l}$
# 
# `ivar` represents: $\frac{1}{\sqrt{\sigma^2 + \epsilon}}$
# 
# `zhat` represents: $\hat{z_l}$
# 
# `q` represents: $q_l$

# In[ ]:


eps = 1e-6 # 𝜖
mu = np.mean(z, axis = 0) # 𝜇
var = np.var(z, axis=0) # 𝜎^2
zmu = z - mu # z - 𝜇
ivar = 1 / np.sqrt(var + eps) # 𝜎𝑖𝑛𝑣
zhat = zmu * ivar 
q = gamma*zhat + beta # ql


# In[ ]:


q


# In[ ]:


mu


# In[ ]:


var


# We will save some of these variables in `cache` as they will be used in backpropagation

# In[ ]:


cache = (gamma, beta, zmu, ivar, zhat)


# > **Note:** During training we also keep an exponentially decaying running value of the mean and variance of each feature, and these averages are used to normalize data at test-time. At each timestep we update the running averages for mean and variance using an exponential decay based on the `momentum` parameter:
#   
# ```
# running_mean = momentum * running_mean + (1 - momentum) * sample_mean
# running_var = momentum * running_var + (1 - momentum) * sample_var
# ```

# **Test-time forward pass for batch normalization**
# 
# We use the running mean and variance to normalize the incoming test data ($z_t$), then scale and shift the normalized data using gamma ($\gamma$) and beta ($\beta$) respectively. Output is stored in $q_t$
# 
# ```
# zt_hat = (zt - running_mean) / np.sqrt(running_var + eps)
# qt = gamma * zt_hat + beta
# ```

# **Backpropagation**

# This `dq` variable below represents $\frac{\partial J}{\partial q_l}$

# In[ ]:


np.random.seed(24)
dq = np.random.randn(m,d)
dq


# `dgamma` represents: $\frac{\partial J}{\partial \gamma}$
# 
# `dbeta` represents: $\frac{\partial J}{\partial \beta}$
# 
# `dzhat` represents: $\frac{\partial J}{\partial \hat{z_l}}$
# 
# `dvar` represents: $\frac{\partial J}{\partial \sigma^2}$
# 
# `dmu` represents: $\frac{\partial J}{\partial \mu}$
# 
# `dz` represents: $\frac{\partial J}{\partial z_l}$
# 
# 

# In[ ]:


dgamma = np.sum(dq * zhat, axis=0)
dbeta = np.sum(dq, axis=0)
dzhat = dq * gamma
dvar = np.sum(dzhat * zmu * (-.5) * (ivar**3), axis=0)
dmu = np.sum(dzhat * (-ivar), axis=0)
dz = dzhat * ivar + dvar * (2/m) * zmu + (1/m)*dmu


# In[ ]:


dgamma


# In[ ]:


dbeta


# In[ ]:


dz

