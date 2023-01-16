#!/usr/bin/env python
# coding: utf-8

# # 2.13. Shortcut to calculate forward pass and backpropagation across layers
# 
# Since we have only `five different operations` (mentioned below; from whatever we have learned till now) that can performed on the input ($X$) to get a certain output ($Z$), so, the following rule can help us evaluate the forward pass and backpropagation error through that particular operation. This is same as obtaining forward pass and backpropagation through the computational graphs. Let me explain.
# 
# > **Note:** $Q^T$ denotes the transpose of the matrix $Q$ and $\sum_c$ denotes sum along the columns (i.e. sum of column-1 then sum of column-2 and so on) to get a vector of length same as number of columns.

# **1. Matrix Multiplication $(*)$**
# 
# Let $(*)$ denote the matrix multiplication between Input $X$ of size $(m,d)$ and the parameter of this blackbox (Matrix Multiplication operation) $W$ of size $(d,h)$ and let the Output be $Z$, whose size will be $(m,h)$. 
# 

# ![](images/mat_mul.png)

# * **Forward Propagation**
# 
# $$Z_{(m,h)} = X_{(m,d)} * W_{(d,h)}$$
# 
# * **Backpropagation**
# 
#   * **Output**
# 
#     $$dX_{(m,d)} = dZ_{(m,h)} * W^T_{(h,d)}$$
# 
#   * **Parameter**
# 
#     $$dW_{(d,h)} = X^T_{(d,m)} * dZ_{(m,h)}$$

# **2. Addition $(+)$**
# 
# Let $(+)$ denote addition between Input $X$ of size $(m,d)$ and the parameter of this blackbox, $b$ of size $(d,1)$ and let the Output be $Z$, whose size will be $(m,d)$. 
# 

# ![](images/add.png)

# * **Forward Propagation**
# 
# $$Z = X + b^T$$
# 
# * **Backpropagation**
# 
#   * **Output**
# 
#     $$dX = dZ$$
# 
#   * **Parameter**
# 
#     $$db = \sum_c dZ$$

# **3. Activation $f(.)$**
# 
# Let $f(.)$ be the activation function that transforms Input $X$ of size $(m,d)$ to an Output $Z$ of same size as that of $X$. Since, the operation is performed element wise on $X$ (each element of $X$ got transformed into the respective elements of $Z$ through $f(.)$), so let us denote $\odot$ as an element wise multiplication operation. This black box has no parameters.
# 
# > **Note:** $f'(.)$ denotes the derivative of the activation function. We have already explained how to calculate these derivatives [here](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/activation.html) (link to previous chapter)

# ![](images/activation.png)

# * **Forward Propagation**
# 
# $$Z = f(X)$$
# 
# * **Backpropagation**
# 
#   * **Output**
# 
#     $$dX = dZ \odot f'(X)$$
# 
#   * **Parameter**
# 
#     $$\text{None}$$

# **4. Dropout $(\text{DR})$**
# 
# Let $\text{DR}$ denote the dropout operation on Input $X$ of size $(m,d)$ to get an Output $Z$ of same size as that of $X$. We have already calculated the **forward ($Z$) and back propagation ($dX$)** results for dropout [here](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/dropout.html) (link to previous chapter). 

# ![](images/dropout.png)

# **4. Batch Normalization $(\text{BN})$**
# 
# Let $\text{BN}$ denote the Batch Normalization operation on Input $X$ of size $(m,d)$ to get an Output $Z$ of same size as that of $X$. We have already calculated the **forward ($Z$) and back propagation ($dX$)** results for Batch Normalization [here](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/batch_normalization.html) (link to previous chapter). 

# ![](images/batch_norm.png)

# #### Example
# 
# Consider the network shown below (assume that the hidden layer also contains activation function)

# ![](images/neural_nets_architecture_2.png)

# Following the different operations discussed above, we can break this network into series of operations as shown below.

# ![](images/series.png)

# **Forward Propagation**
# 
# $$
# R_1 = XW_1
# $$
# 
# $$
# Z_1 =R_1 + b_1^T
# $$
# 
# $$
# A_1 = f_1(Z_1)
# $$
# 
# $$
# R_2 = A_1W_2
# $$
# 
# $$
# Z_2 =R_2 + b_2^T
# $$
# 
# $$
# A_2 = f_2(Z_2)
# $$
# 
# After this we calculate our cost function $J(W, b)$ and then we perform backpropagation.
# 
# **Backpropagation**
# 
# $dA_2$ can be calculated based on the type of cost function we are using. For example if the cost function is **MSE**, then $dA_2 = A_2-y$ (where $y$ is the target variable).
# 
# $$
# dZ_2 = dA_2 \odot f'(Z_2)
# $$
# 
# $$
# dR_2 = dZ_2
# $$
# 
# $$
# db_2 = \sum_c dZ_2
# $$
# 
# $$
# dA_1 = dR_2 \hspace{0.1cm} W_2^T
# $$
# 
# $$
# dW_2 = A_1^T \hspace{0.1cm} dR_2
# $$
# 
# $$
# dZ_1 = dA_1 \odot f'(Z_1)
# $$
# 
# $$
# dR_1 = dZ_1
# $$
# 
# $$
# db_1 = \sum_c dZ_1
# $$
# 
# $$
# dX = dR_1 \hspace{0.1cm} W_1^T
# $$
# 
# $$
# dW_1 = X^T \hspace{0.1cm} dR_1
# $$
# 
# Didn't know Backpropagation can be so easy and intuitive
