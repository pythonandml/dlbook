#!/usr/bin/env python
# coding: utf-8

# # 3.4. 4 step process to build a CNN model using PyTorch

# From our previous chapters (including the one where we have coded [CNN model from scratch](https://pythonandml.github.io/dlbook/content/convolutional_neural_networks/cnn_from_scratch.html)), we now have the idea of how CNN works. Today, we will build our very first CNN model using PyTorch (it just takes quite a few lines of code) in just 4 simple steps.
# 
# #### How to build CNN model using PyTorch
# 
# #### Step-1
# 
# Importing all dependencies
# 
# We first import `torch`, which imports **PyTorch**. Then we import `nn`, which allows us to define a neural network module. 
# 
# Next we import the `DataLoader` with the help of which we can feed data into the convolutional neural network (CNN) during training.
# 
# Finally we import `transforms`, which allows us to perform [data pre-processing](https://pythonandml.github.io/dlbook/content/preliminaries/data_preprocessing.html) (link to previous chapter)

# In[1]:


import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms

# show the progress bar while priting
from tqdm import tqdm


# #### Step-2
# 
# Defining the CNN class as a `nn.Module`
# 
# The CNN class replicates the `nn.Module` class. It has two definitions: __init__, or the constructor, and **forward**, which implements the forward pass.
# 
# We create a convolution model using `nn.Conv2d` and a pooling layer using `nn.Maxpool2d`.
# 
# > **Note:** Here `nn.Linear` is similar to the **Dense** class we developed in our scratch model.

# In[2]:


class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        '''
        The first parameter 3 here represents that the image is colored and in RGB format. 
        If it was a grayscale image we would have gone for 1.
        32 is the size of the initial output channel 
        '''
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
            nn.Flatten(), 
            nn.Linear(64*16*16, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.network(x)


# #### Step-3
# 
# Preparing the **CIFAR-10** dataset and compiling the model *(loss function, and optimizer)*.
# 
# ```{note}
# CIFAR-10 is a dataset that has a collection of images of 10 different classes. This dataset is widely used for research purposes to test different machine learning models and especially for computer vision problems.
# ```
# 
# The next code we add involves preparing the **CIFAR-10** dataset.
# 
# We will define the `batch_size` of 100.

# In[3]:


train_dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=1)

test_dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=1)


# Now, we will initialize the CNN model and compile the same by specifying the loss function (categorical crossentropy loss) and the Adam optimizer. 

# In[8]:


# Initialize the CNN
cnn = CNN()

# Define the loss function (CrossEntropyLoss) and optimizer (ADAM)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)


# #### Step-4
# 
# Defining the training loop
# 
# The core part of our runtime code is the training loop. In this loop, we perform the epochs, or training iterations. For every iteration, we iterate over the training dataset, perform the entire forward and backward passes, and perform model optimization.
# 
# 

# In[9]:


# Run the training loop

# 2 epochs at maximum
epochs = 2

for epoch in range(0, epochs): 
  
    # Print epoch
    print("Epoch:", epoch+1, '/', end=' ')
    
    # Set current loss value
    current_loss = 0.0
    
    # Iterate over the DataLoader for training data
    for i, data in enumerate(tqdm(trainloader)):

        # Get inputs
        inputs, targets = data
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Perform forward pass
        outputs = cnn(inputs)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Perform backward pass
        loss.backward()
        
        # Perform optimization
        optimizer.step()
        
        # Print results
        current_loss += loss.item()
    
    print("Training Loss:", current_loss/len(trainloader))
    

# Process is complete.
print('Training process has finished.')


# #### Testing time!

# In[11]:


cnn.eval()
correct = 0                                               
total = 0                                                 
running_loss = 0.0                                 
with torch.no_grad():                                     
    for i, data in enumerate(tqdm(testloader)):                     
        inputs, targets = data                                                           
        outputs = cnn(inputs)                           
        loss = loss_function(outputs, targets)  

        _, predicted = torch.max(outputs.data, 1)         
        
        total += targets.size(0)                           
        correct += (predicted == targets).sum().item()     
        running_loss = running_loss + loss.item()         
accuracy = correct / total
running_loss = running_loss/len(testloader)
print("\nTest Loss:", running_loss)
print("Test Accuracy:", accuracy)


# With more complex model, we can increase the accuracy of CIFAR-10 as much as we want. The main thing is that we have learnt how to build our very first CNN model using PyTorch in just 4 simple steps.
