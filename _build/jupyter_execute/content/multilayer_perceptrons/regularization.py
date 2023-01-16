#!/usr/bin/env python
# coding: utf-8

# # 2.9. Regularization
# 
# ![](images/regularization.png)

# #### Bias - Variance
# 
# Whenever we discuss model prediction, it’s important to understand prediction errors (bias and variance). There is a tradeoff between a model’s ability to minimize bias and variance. The model performance becomes `poor` in the case where we find either too **high bias** or too **high variance**. 
# 
# **What is bias?**
# 
# Bias is the difference between the average prediction of our model and the correct value which we are trying to predict.
# 
# **What is variance?**
# 
# Variance is the variability of model prediction for a given data point or a value which tells us spread of our data.
# 
# #### High Bias
# 
# The case of `high bias` occurs when both the training error and validation error are high and in such case, the learning of the data doesn’t take place for the models and its predictions therefore becomes poor. This is also called **under-fitting**.
# 
# #### High Variance
# 
# In the case of `high variance`, the training error and validation error diverge (that it the the training error starts to decrease whereas the validation error starts increasing). Such case leads to **over fitting** of the data where the model learns too much from the training examples whereas it isn't able to generalize from the training data to new data.
# 
# #### Regularization helps in reducing overfitting
# 
# Now, in order to avoid this over-fitting, feeding more data during training can help improve the model's performance. Then addition of noise to the input data might also help in generalizing the dataset. We can then add regularization parameter to avoid this situation.
# 
# So, regularization is an optimization technique that prevents overfitting.
# 
# **$L_2$ regularization**
# 
# It consists of adding a term in the cost function $J(W,b)$ to minimize as follows:
# 
# $$
# J(W,b) := J(W,b) + \frac{\lambda}{2} \sum_{l} \left \| W_l \right \|_F^2
# $$
# 
# where $\left \| . \right \|_F$ is the **Frobenius norm** and $l$ is the $l^{th}$ layer.
# 
# So, the partial derivative of $J(W,b)$ with respect to $W_l$ will then be updated as follows:
# 
# $$
# \frac{\partial J(W,b)}{\partial W_l} := \frac{\partial J(W,b)}{\partial W_l} + \lambda W_l
# $$
# 
# > **Note**: The partial derivative of $J(W,b)$ with respect to bias $b_l$ will not be effected.
# 
# **$L_1$ regularization**
# 
# It consists of adding a term in the cost function $J(W,b)$ to minimize as follows:
# 
# $$
# J(W,b) := J(W,b) + \frac{\lambda}{2} \sum_{l} \sum_{i} \sum_{j} \left | W_l (i,j) \right |
# $$
# 
# So, the partial derivative of $J(W,b)$ with respect to $W_l$ will then be updated as follows:
# 
# $$
# \frac{\partial J(W,b)}{\partial W_l} := \frac{\partial J(W,b)}{\partial W_l} + \lambda \hspace{0.1cm} \text{sign}(W_l)
# $$
# 
# where
# 
# $$
# \text{sign}(W_l) = \left\{\begin{matrix}
# 1 & W_l>0\\ 
# -1 & W_l<0
# \end{matrix}\right.
# $$
# 
# > **Note**: The partial derivative of $J(W,b)$ with respect to bias $b_l$ will not be effected.
# 
# **Early stopping**
# 
# This technique is quite simple and consists of stopping the iterations around the area when training loss​ and validation loss​ start diverging. As in the above image, the training must be stopped near 2000 iterations.
# 
