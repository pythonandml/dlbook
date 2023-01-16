#!/usr/bin/env python
# coding: utf-8

# # 3.2. Basic Architecture of CNN
# 
# Consider a black box (as shown in the image below) named `CNN` which takes batch of images (RGB or black and white) as an input, performs some operations on these images and provides an output (in case of classification, the output will be a batch of vector containing probability values for predicting the target labels of the respective batch). For simplicity, consider only one image for now.
# 
# ![](images/cnn_blackbox.png)
# 
# When we open this black-box (see image below), we find that the **CNN** is made up of three types of layers: 
# * Convolutional layers
# * Pooling layers
# * Fully-connected (FC) layers - same as `MLP`.
# 
# ![](images/cnn_architecture.png)
# 
# There is no limitation for the number of such layers present in the CNN model. Let us go through each layer one by one.
