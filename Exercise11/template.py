#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 19:36:21 2022

@author: janik
"""

import torch as tc
import rnn

tc.autograd.set_detect_anomaly(True)

data = tc.load('lorenz63.pt')

''' TASK 1a: Plot the first 500 time steps of the data '''

''' TASK 1b: Choose model and training parameters, initialize and train the model '''

''' TASK 1c: Plot a generated sequence against a true one. '''

''' TASK 1d: Why does this not work?? '''

''' TASK 1e: Calculate the MSE of a generated sequence against a true one. Will you have the lowest in class? :) '''
