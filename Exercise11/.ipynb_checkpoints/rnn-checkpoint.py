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
import time


def plot3D(data, ax=None):
    ''' Makes a 3D plot of data with dimensions (time x 3)'''
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') ######
    ax.plot(data[:,0].detach().numpy(), data[:,1].detach().numpy(), data[:,2].detach().numpy()) ######
    return ax
    
def plot_generated(data, model):
    ''' Generates a time series of the same length as data and plots them against each other '''
    h0 = tc.randn((1, model.latent_dim))
    gen = model.generate(data.shape[0], data[:1], h0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = plot3D(data, ax)
    ax = plot3D(gen, ax)
    return ax

class latent_RNN(nn.Module):
    def __init__(self, obs_dim, latent_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.RNN = nn.RNN(obs_dim, latent_dim, nonlinearity='relu')
        self.observation_model = nn.Linear(latent_dim, obs_dim)        

    def forward(self, time_series, h0):
        rnn_output, h = self.RNN(time_series, h0)
        obs_output = self.observation_model(rnn_output)
        return obs_output, h
    
    def generate(self, T, x0, h0):
        prediction = tc.zeros((T, self.obs_dim))
        x = x0
        h = h0
        for t in range(T):
            prediction[t] = x.squeeze()
            x, h = self(x, h)
        return prediction
   

def train(model, data, epochs, seq_per_epoch, seq_length, L1_reg_strength, 
          verbose=True, plot_loss=True):
    ''' Automatically uses MSE Loss, Adam Optimizer and L1 regularization '''
    
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    model.train()
    
    loss_over_time = []
    t0 = time.time()
    for i in range(epochs):
        optimizer.zero_grad()
        
        random_indices = tc.randperm(data.shape[0] - seq_length)[:seq_per_epoch]
        x = tc.cat([data[r:r+seq_length].unsqueeze(1) for r in random_indices], dim=1)
        y = tc.cat([data[r+1:r+seq_length+1].unsqueeze(1) for r in random_indices], dim=1)
        h0 = tc.randn((1, seq_per_epoch, model.latent_dim))
        
        output, _ = model(x, h0)        
        
        model_loss = loss_function(output, y)
        reg_loss = tc.tensor(0.)
        if L1_reg_strength > 0:
            for name, param in model.named_parameters():
                if 'bias' in name:
                    continue
                reg_loss += param.abs().mean()
            reg_loss *= L1_reg_strength
        epoch_loss = model_loss + reg_loss
        
        epoch_loss.backward()
        optimizer.step()
        if i % 50 == 0 and i>0: ######
            t1 = time.time()
            epochs_per_sec = 10/(t1 - t0) 
            if verbose:
                print(f"Epoch: {i} loss {epoch_loss.item()} @ {epochs_per_sec} epochs per second")
            loss_over_time.append((model_loss.item(), reg_loss.item()))
            t0 = t1
            
    if plot_loss:
        plt.figure()
        plt.plot(tc.tensor(loss_over_time))
        plt.yscale('log')
        plt.legend(('model', 'regul'))
        plt.title('Loss')
        #plt.savefig('Losses', dpi=500) ######
        
    return model, loss_over_time
        
