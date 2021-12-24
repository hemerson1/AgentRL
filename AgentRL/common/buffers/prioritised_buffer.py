#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:13:56 2021

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
# https://github.com/rlcode/per

# TODO: consider a more permenant fix for conversion error
# TODO: look for possible opitmisations

class prioritised_replay_buffer(base_buffer):
    
    def __init__(self, max_size=10_000, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        
        # A warning is appearing saying that list -> tensor conversion is slow
        # However changing to list -> numpy -> tensor is much slower
        warnings.filterwarnings("ignore", category=UserWarning) 
        
        # Initialise the buffer
        self.tree = binary_sum_tree(max_size=max_size)
        
        # Set the buffer parameters
        self.max_size = max_size        
        
        # specify the buffer name
        self.buffer_name = 'prioritised'

        # Define the hyperparameters
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
        # small +ve constant which prevents edge cases from not being visited
        # once their error is zero
        self.epsilon = 0.01
        
    def reset(self):
        
        # reinitialise the sum tree 
        self.tree = binary_sum_tree(max_size=self.max_size)     

    def push(self, error, state, action, next_state, reward, done):
        
        # batch values together
        sample = (state, action, next_state, reward, done)
        
        # calculate the sample priority
        p = self.get_priority(error)
        
        # create the sample and add it to the tree
        self.tree.add(p, sample) 
        
                        
    def sample(self, batch_size, device='cpu'):   
                
        # divide the tree into segments
        segment = self.tree.total() / batch_size
        
        # update beta incremntally until it is 1
        self.beta = min(1., self.beta + self.beta_increment_per_sampling)
        
        # generate a range of samples
        samples = [random.uniform(segment * i, segment * (i + 1)) for i in range(batch_size)]
        
        # get indexes, priorities and data from tree search
        outputs = [self.tree.get(sample) for _, sample in enumerate(samples)]
        idxs, priorities, batch = zip(*outputs)
        
        # calculate importance sampling weights
        tree_total, tree_size = self.tree.total(), self.tree.current_size        
        is_weights = [tree_size * ((priority / tree_total) ** (-self.beta)) for _, priority in enumerate(priorities)]
        is_weights_max = max([is_weights])[0]        
        is_weights = [is_weight / is_weights_max for _, is_weight in enumerate(is_weights)]
        
        # convert the batch into an appropriate tensor form    
        state, action, next_state, reward, done = map(torch.tensor, zip(*batch))
        
        # run the tensors on the selected device
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        
        # convert importance sampling weights to tensor form
        is_weights = torch.FloatTensor(is_weights)
        is_weights - is_weights.to(device)
        
        # make all the tensors 2D
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        
        # check the dimension of the action and convert to 2D
        if len(action.size()) == 1: 
            action = action.unsqueeze(1)
        
        # repackage the batch
        batch_sample = (state, action, next_state, reward, done)

        return batch_sample, idxs, is_weights 
    
    def get_length(self):
        return self.tree.current_size

    def get_priority(self, error):
        return (error + self.epsilon) ** self.alpha       
    
    def update(self, idx, error):
        
        # get the proprity 
        p = self.get_priority(error)
        self.tree.update(idx, p)         
            
            
class binary_sum_tree:
    
    def __init__(self, max_size):
        
        # define the size
        self.max_size = max_size
        self.current_size = 0
        
        # initialise the tree and the data storage
        self.tree = [0] * (2 * max_size - 1) 
        self.data = [0] * max_size 
        self.write = 0
        
        
    # store priority and sample
    def add(self, priority, data):
        
        # get the data index (fill from the end)
        idx = self.write + self.max_size - 1
        
        # set the data
        self.data[self.write] = data
        self.update(idx, priority)
        
        # update the end
        self.write += 1
        
        # start overwriting initial values
        if self.write >= self.max_size:
            self.write = 0
        
        # update the current_size until max
        if self.current_size < self.max_size:
            self.current_size += 1
            

    # update priority
    def update(self, idx, priority):
        
        # get the change in priority
        change = priority - self.tree[idx]
        
        # update the priority at that index
        self.tree[idx] = priority
        
        # update all values to root node
        self.propagate(idx, change)
        
        
    # update to the root node
    def propagate(self, idx, change):
        
        # get the parent node
        parent = (idx - 1) // 2
        
        # add sum to parent node
        self.tree[parent] += change
        
        # if this isn't the root node -> recursion
        if parent != 0:
            self.propagate(parent, change)
            
            
    # get priority and sample
    def get(self, sample):
        
        # get the index
        idx = self.retrieve(0, sample)
        
        # get the accompanying data
        dataIdx = idx - self.max_size + 1

        return (idx, self.tree[idx], self.data[dataIdx])
    

    # find sample on leaf node
    def retrieve(self, idx, sample):
        
        # get the child nodes of the current node
        left = 2 * idx + 1
        right = 2 * idx + 2

        # if this is a leaf node      
        if left >= len(self.tree):
            return idx
        
        # if priority us less than left node follow left node
        if sample <= self.tree[left]:
            return self.retrieve(left, sample)
        
        # follow the right node
        else:
            return self.retrieve(right, sample - self.tree[left])
        
    
    def total(self):
        
        # get the root sum
        return self.tree[0]
    
# TESTING ###################################################
        
if __name__ == '__main__':
    
    buffer = prioritised_replay_buffer(max_size=100_000)
    
    # test the appending to the array    
    tic = time.perf_counter()
    
    for i in range(100_005):
        
        state = [random.randint(0, 10), random.randint(0, 10)]
        next_state = random.randint(0, 10)
        action = random.randint(0, 10)
        reward = random.randint(0, 10)
        done = False
        
        buffer.push(error=1, state=state, action=action, next_state=state, reward=reward, done=done)
        
        if i > 100_000:
            pass
            # print(buffer.buffer[-1])
        
    toc = time.perf_counter()
    print('Appending took {} seconds'.format(toc - tic))    
    print('Final buffer length: {}'.format(buffer.get_length()))  
    
    # test the sampling from the array
    tic_1 = time.perf_counter()
    
    for i in range(10_000):
        
        batch, idxs, is_weight = buffer.sample(batch_size=32)
        state, action, reward, next_state, done = batch        
        
        if i % 1_000 == 0:
            # print(action)
            # print(type(action))
            # print(action.shape)
            # print('------------')
            pass
        
    toc_1 = time.perf_counter()
    print('Sampling took {} seconds'.format(toc_1 - tic_1))    
    
#################################################################