#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 14:06:47 2021

@author: hemerson
"""

from AgentRL.common.buffers.base import base_buffer

import numpy as np
import random

# TESTING 
import time

# The core structure of this buffer was inspired by:
# https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/common/buffers.py

# TODO: Continue optimising -> how does speed scale with increased batch size

class standard_replay_buffer(base_buffer):
    
    def __init__(self, max_size=10_000):
        
        # Initialise the buffer
        self.buffer = []
        
        # Set the buffer parameters
        self.max_size = max_size

    def push(self, state, next_state, action, reward, done):
        
        # add an extra element to a non-full list
        self.buffer.append(None)
           
        # set the current buffer position to the input
        self.buffer[-1] = (state, next_state, action, reward, done)
        
        # trim list to the max size
        if len(self.buffer) > self.max_size:
            del self.buffer[0]
                        
    def sample(self, batch_size):        
        
        # get a batch and unpack it into its constituents
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))  # stack for each element
        
        return state, action, reward, next_state, done
            
        
    def get_length(self):
        return len(self.buffer)        
        
if __name__ == '__main__':
    
    buffer = standard_replay_buffer(max_size=100_000)
    
    # TESTING ###################################################
    
    # test the appending to the array    
    tic = time.perf_counter()
    
    for i in range(100_005):
        
        state = [random.randint(0, 10), random.randint(0, 10)]
        next_state = random.randint(0, 10)
        action = random.randint(0, 10)
        reward = random.randint(0, 10)
        done = random.randint(0, 10)
        
        buffer.push(state, next_state, action, reward, done)
        
        if i > 100_000:
            print(buffer.buffer[-1])
        
    toc = time.perf_counter()
    print('Appending took {} seconds'.format(toc - tic))    
    print('Final buffer length: {}'.format(buffer.get_length()))  
    
    # test the sampling from the array
    tic_1 = time.perf_counter()
    
    for i in range(10_000):
        
        buffer.sample(batch_size=64)
        
    toc_1 = time.perf_counter()
    print('Sampling took {} seconds'.format(toc_1 - tic_1))    
    
#################################################################

    