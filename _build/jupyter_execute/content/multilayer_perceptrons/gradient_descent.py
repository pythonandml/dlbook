#!/usr/bin/env python
# coding: utf-8

# # 2.8. Gradient Descent
# 
# We have already seen the gradient descent and **update law** in [terminologies : part-2](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/terminologies_part_2.html) (link to previous chapter). Just for the sake of completeness, let us quickly revisit the same.
# 
# #### Update law
# 
# Gradient Descent iteratively calculates the next value of a variable ($p_{n+1}$) using gradient of that variable ($\frac{\partial J}{\partial p_n}$) at the current iteration, scales it (by a learning rate, $\eta$) and subtracts obtained value from the current position (also called as taking a step). It subtracts the value because we want to minimise the function (to maximise it would be adding). This process can be written as:
# 
# $$
# p_{n+1} = p_n - \eta \frac{\partial J}{\partial p_n}
# $$
# 
# There’s an important parameter $\eta$ which *scales* the gradient and thus controls the step size. In machine and deep learning, it is called **learning rate** and have a strong influence on performance.
# 
# #### Mini-batch gradient descent
# 
# This technique consists of dividing the training set to [batches](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/terminologies_part_2.html#batch-size) (link to previous chapter):
# 
# Suppose we are given the batches {$(X_1, y_1)$, $(X_2, y_2)$, ... ,$(X_{N_b}, y_{N_b})$} where $N_b$ is the `number of batches`.
# 
# * for $t=1,2,...N_b$
#     * Carry out forward propagation on $X_t$
#     * Compute cost function normalized on the size of batch
#     * Carry out Backpropagation using $(X_t, y_t, \hat{y_t})$
#     * Update weight $W$ and $b$
# 
# > **Note:** In the case where there is only one data sample (selected randomly) in the batch, the algorithm is called **stochastic gradient descent**
# 
# #### Gradient descent with momentum
# 
# Gradient descent with momentum is a variant of gradient descent which includes the notion of `momentum`. 
# 
# It is a method which helps accelerate gradients vectors in the right directions, thus leading to faster converging.
# 
# The algorithm is as follows:
# 
# For any layer $l=(1,2,...L)$
# 
# * Initialize $v_{dW_l}=0$ (size same as $dW_l$) and $v_{db_l}=0$ (size same as $db_l$)
# 
# * On iteration $k$
#     * Compute $dW_l$ and $db_l$ on current mini batch
#     
#     $$v_{dW_l} = \alpha \hspace{0.1cm} v_{dW_l} + (1-\alpha) \hspace{0.1cm}dW_l$$
#     
#     $$v_{db_l} = \alpha \hspace{0.1cm} v_{db_l} + (1-\alpha) \hspace{0.1cm}db_l$$
#     
#     * Update the parameters
#     
#         $$W_l := W_l - \eta \hspace{0.1cm} v_{dW_l}$$
#         
#         $$b_l := b_l - \eta \hspace{0.1cm} v_{db_l}$$
# 
# The hyper-parameter $\alpha$ is called the **momentum**. In deep learning, most practitioners set the value of $\alpha=0.9$ without attempting to further tune this hyperparameter (i.e., this is the default value for momentum in many popular deep learning packages). 
# 
# #### Root Mean Square prop - RMSProp 
# 
# It is very similar to gradient descent with momentum, the only difference is that it includes the second-order momentum instead of the first-order one, plus a slight change on the parameter's update:
# 
# The algorithm is as follows:
# 
# For any layer $l=(1,2,...L)$
# 
# * Initialize $S_{dW_l}=0$ (size same as $dW_l$) and $S_{db_l}=0$ (size same as $db_l$)
# 
# * On iteration $k$
#     * Compute $dW_l$ and $db_l$ on current mini batch
#     
#     $$S_{dW_l} = \alpha \hspace{0.1cm} S_{dW_l} + (1-\alpha) \hspace{0.1cm}dW_l^2$$
#     
#     $$S_{db_l} = \alpha \hspace{0.1cm} S_{db_l} + (1-\alpha) \hspace{0.1cm}db_l^2$$
#     
#     * Update the parameters
#     
# $$W_l := W_l - \frac{\eta}{\sqrt{S_{dW_l}} + \epsilon} \hspace{0.1cm} dW_l$$
# 
# $$b_l := b_l - \frac{\eta}{\sqrt{S_{db_l}} + \epsilon} \hspace{0.1cm} db_l$$
# 
# 
# #### Adam
# 
# Adam (adaptive learning rate optimization) can be seen as a combination of RMSprop and gradient descent with momentum. The main idea is to avoid oscillations during optimization by accelerating the descent in the right direction.
# 
# The algorithm is as follows:
# 
# For any layer $l=(1,2,...L)$
# 
# * Initialize $v_{dW_l}=0$, $v_{db_l}=0$, $S_{dW_l}=0$ and $S_{db_l}=0$
# 
# * On iteration $k$
#     * Compute $dW_l$ and $db_l$ on current mini batch
#     
#     * Momentum
#         
#         $v_{dW_l} = \alpha_1 \hspace{0.1cm} v_{dW_l} + (1-\alpha_1) \hspace{0.1cm}dW_l$
#         
#         $v_{db_l} = \alpha_1 \hspace{0.1cm} v_{db_l} + (1-\alpha_1) \hspace{0.1cm}db_l$
#     
#     * RMSProp
#         
#         $S_{dW_l} = \alpha_2 \hspace{0.1cm} S_{dW_l} + (1-\alpha_2) \hspace{0.1cm}dW_l^2$
#         
#         $S_{db_l} = \alpha_2 \hspace{0.1cm} S_{db_l} + (1-\alpha_2) \hspace{0.1cm}db_l^2$
#     
#     * Correction
# 
#     $$
#     v_{dW_l} = \frac{v_{dW_l}}{1-\alpha_1^k}
#     $$
# 
#     $$
#     v_{db_l} = \frac{v_{db_l}}{1-\alpha_1^k}
#     $$
# 
#     $$
#     S_{dW_l} = \frac{S_{dW_l}}{1-\alpha_2^k}
#     $$
# 
#     $$
#     S_{db_l} = \frac{S_{db_l}}{1-\alpha_2^k}
#     $$
# 
#     * Update the parameters
#     
# $$W_l := W_l - \frac{\eta}{\sqrt{S_{dW_l}} + \epsilon} \hspace{0.1cm} v_{dW_l}$$
# 
# $$b_l := b_l - \frac{\eta}{\sqrt{S_{db_l}} + \epsilon} \hspace{0.1cm} v_{db_l}$$
# 
# > **Note:** Good default settings for the tested Machine Learning and Deep learning models are $\eta = 0.001$, $\alpha_1 = 0.9$, $\alpha_2 = 0.999$ and $\epsilon = 10^{-8}$.
# 
# #### Learning Rate Decay
# 
# The main objective of the learning rate decay is to slowly reduce the learning rate over time/iterations. There exist many learning rate decay laws, here are some of the most common:
# 
# **Time-Based Decay**
# 
# The mathematical form of time-based decay for learning rate $\eta_t$ is:
# 
# $$
# \eta_{t+1} = \frac{\eta_0}{1+Kt}
# $$
# 
# where $t$ is the `iteration number` and $K$ is the hyper-parameter called `Decay rate`. Let $E$ be the total number of epochs, then usually we take $K=\frac{\eta_0}{E}$ where $\eta_0$ is the initial learning rate.
# 
# **Step Decay**
# 
# Step decay drops the learning rate by a factor of every few epochs. For example, let’s suppose our initial learning rate is $\eta_0 = 0.01$.
# 
# After 10 epochs we drop the learning rate to $\eta = 0.005$.
# 
# After another 10 epochs (i.e., the 20th total epoch), $\eta$ is dropped by a factor of 0.5 again, such that $\eta = 0.0025$, etc.
# 
# The mathematical form of step decay is:
# 
# $$
# \eta_{e+1} = \eta_0 \hspace{0.12cm} F^{\left \lfloor \frac{1+e}{D} \right \rfloor}
# $$
# 
# Where $\eta_0$ is the initial learning rate, $F$ is the factor value controlling the rate in which the learning date drops, $D$ is the “Drop every” epochs value, $e$ is the current epoch and $\lfloor x \rfloor$ is the $\text{floor(x)}$
# 
# The larger our factor $F$ is, the slower the learning rate will decay and conversely, the smaller the factor $F$, the faster the learning rate will decay.
# 
# **Exponential Decay**
# 
# Another common schedule is exponential decay. It has the mathematical form: 
# 
# $$
# \eta_{t+1} = \eta_0 * e^{−kt}
# $$
# 
# where $k$ is `hyper-parameter` and $t$ is the `iteration number`.
# 
# 

# #### Problems related to Gradients
# 
# **Vanishing gradient**
# 
# As the number of layers in the neural networks increase, the gradient value (used during back propagation) decreases and eventually it tends to zero. This is called `vanishing gradient problem`. The result is that the weights of the model now stops updating and model cannot learn further.
# 
# This mostly happens in the case *when number of layers are too high* or the activation function used in the model is `sigmoid` or `tanh`. 
# 
# The remedy for this problem is to use `ReLU activation` function or initialize the parameters in such a way that the weight value doesn’t become zero.
# 
# **Exploding gradient**
# 
# In contrast to the vanishing gradient problem, in exploding gradient problem, the gradients instead of vanishing, accumulates and results in a very large value (tending to infinity) during training. 
# 
# This makes the model unstable and leads to a poor prediction reporting nan values (**n**ot **a** **n**umber) most of the time.
# 
# There are methods to fix exploding gradients, which include `gradient clipping` (where we clip the gradient to certain range), `data normalization`, `weight regularization`, etc.
