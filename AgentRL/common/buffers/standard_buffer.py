#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:06:47 2021

@author: hemerson
"""

""" 
standard_replay_buffer - A simple replay buffer storing samples of data and 
                         then returning a random batch 

"""

from AgentRL.common.buffers.base import base_buffer

import random
import torch
import warnings

# TESTING 
import time

# The core structure of this buffer was inspired by:
# https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/common/buffers.py

# TODO: Continue optimising -> how does speed scale with increased batch size

class standard_replay_buffer(base_buffer):
    
    def __init__(self, max_size=10_000):
        
        # TODO: consider a more permenant fix for conversion error
        
        # A warning is appearing saying that list -> tensor conversion is slow
        # However changing to list -> numpy -> tensor is much slower
        warnings.filterwarnings("ignore", category=UserWarning) 
        
        # Initialise the buffer
        self.buffer = []
        
        # Set the buffer parameters
        self.max_size = max_size        
        
    def reset(self):
        
        # empty the buffer
        self.buffer = []        

    def push(self, state, action, reward, next_state, done):
        
        # add the most recent sample
        self.buffer.append((state, action, reward, next_state, done))
           
        # trim list to the max size
        if len(self.buffer) > self.max_size:
            del self.buffer[0]
                        
    def sample(self, batch_size, device='cpu'):        
        
        # get a batch and unpack it into its constituents
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(torch.tensor, zip(*batch))
        
        # run the tensors on the selected device
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        
        # make all the tensors 2D
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        
        # check the dimension of the action and convert to 2D
        if len(action.size()) == 1: 
            action = action.unsqueeze(1)
        
        return state, action, reward, next_state, done            
        
    def get_length(self):
        return len(self.buffer)        
    
    
# TESTING ###################################################
        
if __name__ == '__main__':
    
    buffer = standard_replay_buffer(max_size=100_000)
    
    # test the appending to the array    
    tic = time.perf_counter()
    
    for i in range(100_005):
        
        state = [random.randint(0, 10), random.randint(0, 10)]
        next_state = random.randint(0, 10)
        action = random.randint(0, 10)
        reward = random.randint(0, 10)
        done = False
        
        buffer.push(state, next_state, action, reward, done)
        
        if i > 100_000:
            print(buffer.buffer[-1])
        
    toc = time.perf_counter()
    print('Appending took {} seconds'.format(toc - tic))    
    print('Final buffer length: {}'.format(buffer.get_length()))  
    
    # test the sampling from the array
    tic_1 = time.perf_counter()
    
    for i in range(10_000):
        
        state, action, _, _, done = buffer.sample(batch_size=32)
        
        if i % 1_000 == 0:
            print(action)
            print(type(action))
            print(action.shape)
            print('------------')
        
    toc_1 = time.perf_counter()
    print('Sampling took {} seconds'.format(toc_1 - tic_1))    
    
#################################################################

    