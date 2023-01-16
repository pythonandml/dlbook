#!/usr/bin/env python
# coding: utf-8

# # 2.5. Forward propagation
# 
# As the name suggests, in forward propagation, the input data $X$ is fed in the forward direction through the network. Each hidden layer accepts the input data, processes it as per the activation function and passes to the successive layer.
# 
# > Keeping note of the [notations used in terminologies part-1](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/terminologies_part_1.html#notations-to-be-used) (link to previous chapter), forward propagation is a simple process.
# 
# For each layer $l=1,2,...L$, we compute the weighted sum $z_l$ and its activation $a_l$ as follows (vectorized form for computation efficiency)
# 
# $$
# z_l = a_{l-1}W_l + b_l^T
# $$
# 
# $$
# a_l = f_l(z_l)
# $$
# 
# where $b_l^T$ is the transpose of $b_l$ and $f_l(x)$ is the activation function used in the $l^{th}$ layer.
