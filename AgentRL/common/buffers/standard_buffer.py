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

from AgentRL.common.buffers import base_buffer

import random
import torch
import warnings
import numpy as np

# TESTING 
import time

# The core structure of this buffer was inspired by:
# https://github.com/quantumiracle/Popular-RL-Algorithms/blob/master/common/buffers.py

# The multi-step implementation was guided by:
# https://github.com/andri27-ts/Reinforcement-Learning/blob/master/Week3/buffers.py

# TODO: Continue optimising -> how does speed scale with increased batch size
# TODO: consider a more permenant fix for conversion error

class standard_replay_buffer(base_buffer):
    
    def __init__(self, max_size=10_000, seed=None):
        
        # A warning is appearing saying that list -> tensor conversion is slow
        # However changing to list -> numpy -> tensor is much slower
        warnings.filterwarnings("ignore", category=UserWarning) 
        
        # Initialise the buffer
        self.buffer = []
        
        # Set the buffer parameters
        self.max_size = max_size 
        
        # specify the buffer name
        self.buffer_name = 'default'
        
        # set the seed
        self.seed = seed
        
        # reset to default
        self.reset()
        
    def reset(self):
        
        # reset the seeding
        if type(self.seed) == int:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)  
        
        # empty the buffer
        self.buffer = []
        self.current_size = 0        

    def push(self, state, action, next_state, reward, done):
        
        # add the most recent sample
        self.buffer.append((state, action, next_state, reward, done))
        self.current_size +=1
           
        # trim list to the max size
        if len(self.buffer) > self.max_size:
            del self.buffer[0]
            self.current_size -=1
                        
    def sample(self, batch_size, device='cpu', multi_step=1, gamma=0.99):        
        
        # get a batch and unpack it into its constituents
        if multi_step > 1:
            
            # get a list of sample indexes
            indexes = random.sample(range(0, self.current_size - (multi_step - 1)), batch_size)	
            
            # get the 0th state
            batch = [self.buffer[idx] for idx in indexes]
            state, action, _, _, _ = map(torch.tensor, zip(*batch))
            
            # get the dones and rewards
            next_state_indexes, done, reward = [], [], []         
            for idx in indexes:
                
                # get a range n_steps ahead for this sample
                index_range = self.buffer[idx : idx + (multi_step - 1)]                
                n_step_sample = [(index[-2], index[-1]) for index in index_range]
                
                # unpackage the rewards and dones
                rewards, dones = map(list, zip(*n_step_sample))                
                
                # cut the reward to the appropriate length
                try: 
                    done_index = dones.index(True) + 1 
                    done_val = True
                
                except: 
                    done_index = len(rewards)
                    done_val = False
                
                # discount the rewards
                rewards = rewards[:done_index]                
                reward_val = sum([reward * (gamma ** idx) for idx, reward in enumerate(rewards)])
                
                # get the index for the next state before termination
                next_state_indexes.append(done_index + idx)
                reward.append(reward_val)
                done.append(done_val)
            
            # get the nth state
            next_batch = [self.buffer[idx] for idx in next_state_indexes]
            _, _, next_state, _, _ = map(torch.tensor, zip(*next_batch))
            
            # convert to tensor
            done = torch.tensor(done)
            reward = torch.tensor(reward)
            
            
        else:
            batch = random.sample(self.buffer, batch_size)            
            state, action, next_state, reward, done = map(torch.tensor, zip(*batch))
        
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
        
        return state, action, next_state, reward, done            
        
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
            # print(buffer.buffer[-1])
            pass
        
    toc = time.perf_counter()
    print('Appending took {} seconds'.format(toc - tic))    
    print('Final buffer length: {}'.format(buffer.get_length()))  
    
    # test the sampling from the array
    tic_1 = time.perf_counter()
    
    for i in range(10_000):
        
        state, action, _, _, done = buffer.sample(batch_size=32, multi_step=1)
        
        if i % 1_000 == 0:
            # print(action)
            # print(type(action))
            # print(action.shape)
            # print('------------')
            pass
        
    toc_1 = time.perf_counter()
    print('Sampling took {} seconds'.format(toc_1 - tic_1))    
    
#################################################################

    