#!/usr/bin/env python
# coding: utf-8

# # 2.15. 4 step process to build MLP model using PyTorch

# From our previous chapters (including the one where we have coded [MLP model from scratch](https://pythonandml.github.io/dlbook/content/multilayer_perceptrons/neural_networks_mlp_scratch_best.html)), we now have the idea of how MLP works. Today, we will build our very first MLP model using PyTorch (it just takes quite a few lines of code) in just 4 simple steps.
# 
# #### How to build MLP model using PyTorch
# 
# #### Step-1
# 
# Importing all dependencies
# 
# We first import `torch`, which imports **PyTorch**. Then we import `nn`, which allows us to define a neural network module. 
# 
# Next we import the `DataLoader` with the help of which we can feed data into the neural network (MLP) during training.
# 
# Finally we import `transforms`, which allows us to perform [data pre-processing](https://pythonandml.github.io/dlbook/content/preliminaries/data_preprocessing.html) (link to previous chapter)

# In[ ]:


import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# show the progress bar while priting
from tqdm import tqdm


# #### Step-2
# 
# Defining the MLP class as a `nn.Module`
# 
# The MLP class replicates the `nn.Module` class. It has two definitions: __init__, or the constructor, and **forward**, which implements the forward pass.
# 
# We create a sequential model using `nn.Sequential` where we will add layers of MLP one by one (in the form of a stack) and store it in variable **self.layers**. We also add `nn.Flatten()` which converts the 3D image representations (width, height and channels) into 1D format.
# 
# Our model layers are *three densely-connected layers with Linear and ReLU activation functions*
# 
# > **Note:** Here `nn.Linear` is similar to the **Dense** class we developed in our scratch model.

# In[ ]:


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Flatten(),
      # input shape = 28*28
      # neurons in first dense layer = 64
      nn.Linear(28*28, 64),
      # relu activation
      nn.ReLU(),
      # 64 = neurons in first dense layer
      # 32 = neurons in second dense layer
      nn.Linear(64, 32),
      nn.ReLU(),
      # 32 = neurons in second dense layer
      # 10 = neurons in output layer (number of classes)
      nn.Linear(32, 10)
    )


  def forward(self, x):
    '''Forward pass'''
    return self.layers(x)


# #### Step-3
# 
# Preparing the **MNIST** dataset and compiling the model *(loss function, and optimizer)*.
# 
# The next code we add involves preparing the **MNIST** dataset. The dataset contains 10 classes and has 70,000 28 by 28 pixel images, with 7000 images per class.
# 
# We will define the `batch_size` of 100.

# In[ ]:


train_dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=True)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=1)

test_dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor(), train=False)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=1)


# Now, we will initialize the MLP model and compile the same by specifying the loss function (categorical crossentropy loss) and the Adam optimizer. 

# In[ ]:


# Initialize the MLP
mlp = MLP()

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)


# #### Step-4
# 
# Defining the training loop
# 
# The core part of our runtime code is the training loop. In this loop, we perform the epochs, or training iterations. For every iteration, we iterate over the training dataset, perform the entire forward and backward passes, and perform model optimization.
# 
# 

# In[ ]:


# Run the training loop

# 5 epochs at maximum
epochs = 5 

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
        outputs = mlp(inputs)
        
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

# In[ ]:


mlp.eval()
correct = 0                                               
total = 0                                                 
running_loss = 0.0                                 
with torch.no_grad():                                     
    for i, data in enumerate(tqdm(testloader)):                     
        inputs, targets = data                                                           
        outputs = mlp(inputs)                           
        loss = loss_function(outputs, targets)  

        _, predicted = torch.max(outputs.data, 1)         
        
        total += targets.size(0)                           
        correct += (predicted == targets).sum().item()     
        running_loss = running_loss + loss.item()         
accuracy = correct / total
running_loss = running_loss/len(testloader)
print("\nTest Loss:", running_loss)
print("Test Accuracy:", accuracy)

