#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:01:32 2022

@author: janik
"""
import torch as tc
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

tc.autograd.set_detect_anomaly(True)

# TASK 2: set hidden size
hidden_size = ?
epochs = 500

learning_rate = 0.001

# TASK 4: You need these variables for mini-batching
# seq_length = 40
# seq_per_epoch = 1

data = tc.load('sinus.pt')
observation_size = data.shape[1]

plt.plot(data)
plt.savefig('inputData', dpi=500)
plt.close()

class latent_RNN(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        # TASK 2: Implement the RNN here

    def forward(self, time_series, h0):
        # TASK 2: Implement the forward loop. The output should be observation x and next hidden state h
        return obs_output, h

model = latent_RNN(observation_size, hidden_size)

# TASK 3: Change the parameters of the optimizer, and replace SGD by Adam
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
loss_function = nn.MSELoss()

def train():
    for i in range(epochs):
        h0 = tc.randn((1, hidden_size))
        x = data([:-1])  #x_0:T-1
        y = data([1:])   #x_1:T
        
        # TASK 4: Here you need to do the mini-batching
        
        optimizer.zero_grad()               
        output, _ = model(x, h0)            
        epoch_loss = loss_function(output, y)
        epoch_loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch: {} loss {}".format(i, epoch_loss.item()))

train()

with tc.no_grad():
    h = tc.zeros((1, hidden_size))
    predictions = tc.zeros((6*data.shape[0], observation_size))
    input_ = data[0:1]
    for i in range(6*data.size(0)):
        pred, h = model(input_, h)
        input_ = pred
        predictions[i] = pred
    
    plt.plot(data[1:])
    plt.plot(predictions)
    plt.savefig('Predictions', dpi=500)

