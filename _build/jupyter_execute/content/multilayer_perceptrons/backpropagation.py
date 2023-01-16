#!/usr/bin/env python
# coding: utf-8

# # 2.6. Back Propagation
# 
# The backpropagation is the second step of the learning, which consists of injecting the error committed in the forward propagation phase (error while making predictions because the parameters are not completely trained yet) into the network in the reverse direction (from output layer to input layer) and update its parameters to perform better on the next iteration. 
# 
# Today, the backpropagation algorithm is the workhorse of learning in neural networks. At the heart of backpropagation is an expression for the partial derivative of the cost function $J(W, b)$ with respect to weight $W$ (or bias $b$) in the network. The expression tells us how quickly the cost changes when we change the weights and biases.
# 
# Hence, the optimization of the cost function $J(W, b)$ is needed and it is usually performed through a descent method. 
# 
# #### Derivation using Chain rule
# 
# > **Note:** $\odot$ means element wise multiplication (also called **Hadamard product**)
# 
# Since we need to calculate the partial derivative of the [cost function](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/cost_functions.html) $J(W, b)$ (link to previous chapter) with respect to weight $W$ (or bias $b$), we can do it using chain rule.
# 
# Backpropagation can be summarized using four different equations (vectorized form) considering the [notations presented in terminologies part-1](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/terminologies_part_1.html#notations-to-be-used) (link to previous chapter).
# 
# > **Note:** For any layer $l$ ($l=1,2,...,L$), we call the `backpropagation error` ($\delta_l$) in that layer as the partial derivative of the cost function $J(W, b)$ with respect to weighted sum of that layer $z_l$. That is:
# 
# $$
# \delta_l = \frac{\partial J(W, b)}{\partial z_l}
# $$
# 
# **Output layer Backpropagation Error $(\delta_L)$**
# 
# For the output layer ($l=L$), we know from [forward propagation](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/forward_propagation.html) (link to previous chapter) that 
# 
# $$
# z_L = a_{L-1}W_L + b_L^T 
# $$
# 
# $$
# a_L = f_L(z_L)
# $$
# 
# and the cost function $J(W,b)$ is written using $a_L$ and $y$. So, using the chain rule,
# 
# $$
# \delta_L = \frac{\partial J(W, b)}{\partial z_L} = \frac{\partial J(W, b)}{\partial a_L} \odot \frac{\partial a_L}{\partial z_L}
# $$
# 
# $$\boxed{\therefore \delta_L = \frac{\partial J(W, b)}{\partial a_L} \odot f'_L(z_L)}$$
# 
# **Hidden layer Backpropagation Error $(\delta_l)$**
# 
# Now, for the hidden layers ($l=L-1, L-2,...,1$), we know from [forward propagation](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/forward_propagation.html) (link to previous chapter) that 
# 
# $$
# z_l = a_{l-1}W_l + b_l^T 
# $$
# 
# $$
# a_l = f_l(z_l)
# $$
# 
# and
# 
# $$
# z_{l+1} = a_lW_{l+1} + b_{l+1}^T 
# $$
# 
# Therefore, 
# 
# $$
# \delta_l = \frac{\partial J(W, b)}{\partial z_l} = \frac{\partial J(W, b)}{\partial a_l} \odot \frac{\partial a_l}{\partial z_l}
# $$
# 
# We have 
# 
# $$
# \frac{\partial J(W, b)}{\partial a_l} = \frac{\partial J(W, b)}{\partial z_{l+1}} \frac{\partial z_{l+1}}{\partial a_l} 
# $$
# 
# $$
# \therefore \frac{\partial J(W, b)}{\partial a_l} = \delta_{l+1} W_{l+1}^T
# $$
# 
# So,
# 
# $$
# \boxed{\delta_l = (\delta_{l+1} W_{l+1}^T) \odot f'_l(z_l)}
# $$
# 
# **Rate of change of the cost with respect to weights $W_l$**
# 
# Now, the partial derivative of the cost function $J(W, b)$ with respect to weight of that layer $W_l$ will be:
# 
# $$
# \frac{\partial J(W, b)}{\partial W_l} = \frac{\partial J(W, b)}{\partial z_l} \frac{\partial z_l}{\partial W_l}
# $$
# 
# $$
# \boxed{\therefore \frac{\partial J(W, b)}{\partial W_l} = a_{l-1}^T \delta_l}
# $$
# 
# **Rate of change of the cost with respect to bias $b_l$**
# 
# Finally, the partial derivative of the cost function $J(W, b)$ with respect to bias of that layer $b_l$ will be:
# 
# $$
# \frac{\partial J(W, b)}{\partial b_l} = \frac{\partial J(W, b)}{\partial z_l} \frac{\partial z_l}{\partial b_l}
# $$
# 
# $$
# \boxed{\therefore \frac{\partial J(W, b)}{\partial b_l} = \sum \delta_l}
# $$
# 
# #### Equations summary
# 
# 1. **Output layer Backpropagation error** ($\delta_L$)
# 
# $$
# \delta_L = \frac{\partial J(W, b)}{\partial a_L} \odot f'_L(z_L)
# $$
# 
# 2. **Hidden layer Backpropagation error** ($\delta_l$)
# 
# $$
# \delta_l = (\delta_{l+1} W_{l+1}^T) \odot f'_l(z_l)
# $$
# 
# 3. **Rate of change of the cost with respect to weights $W_l$**
# 
# $$
# \frac{\partial J(W, b)}{\partial W_l} = a_{l-1}^T \delta_l
# $$
# 
# 4. **Rate of change of the cost with respect to bias $b_l$**
# 
# $$
# \frac{\partial J(W, b)}{\partial b_l} = \sum \delta_l
# $$
# 
# where $\sum \delta_l$ is summation over all the samples of $\delta_l$. Size of $\delta_l$ is $(m, h_l)$. So, we sum along the column (i.e. sum of column-1 then sum of column-2 and so on) to get a vector of size $(h_l, 1)$ which is same as $b_l$.
# 
# Also, $Q^T$ denotes the transpose of any matrix $Q$. That's it. We now use these gradients to update the parameters weight $W$ and bias $b$.
# 
# 
# 
