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

from AgentRL.common.buffers import base_buffer

import random
import torch
import warnings
import numpy as np

# TESTING 
import time

# The core structure of this buffer was inspired by:
# https://github.com/rlcode/per

# The multi-step implementation was guided by:
# https://github.com/andri27-ts/Reinforcement-Learning/blob/master/Week3/buffers.py

# TODO: consider a more permenant fix for conversion error
# TODO: look for possible opitmisations

class prioritised_replay_buffer(base_buffer):
    
    def __init__(self, max_size=10_000, seed=None, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        
        # A warning is appearing saying that list -> tensor conversion is slow
        # However changing to list -> numpy -> tensor is much slower
        warnings.filterwarnings("ignore", category=UserWarning) 
        
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
        
        # set the seed
        self.seed = seed
        
        self.reset()
        
    def reset(self):
        
        # reset the seeding
        if type(self.seed) == int:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)  
        
        # reinitialise the sum tree 
        self.tree = binary_sum_tree(max_size=self.max_size)    
        
        # set the max priority
        self.max_priority = 1
        
    def push(self, state, action, next_state, reward, done):
        
        # batch values together
        sample = (state, action, next_state, reward, done)
        
        # calculate the sample priority
        p = self.max_priority 
        
        # create the sample and add it to the tree
        self.tree.add(p, sample) 
        
                        
    def sample(self, batch_size, device='cpu', multi_step=1, gamma=0.99): 
                        
        # divide the tree into segments
        segment = self.tree.total() / batch_size
        
        # update beta incrementally until it is 1
        self.beta = min(1., self.beta + self.beta_increment_per_sampling)
        
        # generate a range of samples (need to cap priority at the current max)
        samples = [random.uniform(segment * i, min(segment * (i + 1), self.max_priority)) for i in range(batch_size)]        
        
        # get indexes, priorities and data from tree search
        outputs = [self.tree.get(sample) for _, sample in enumerate(samples)]        
        
        if multi_step > 1:
            
            # get the current batch
            idxs, priorities, batch, data_idxs = zip(*outputs)
            
            # get the dones and rewards
            next_state_indexes, done, reward = [], [], []    
            
            for idx in data_idxs:
                
                # get a range n_steps ahead for this sample
                if idx + (multi_step - 1) >= self.max_size: 
                    index_range = self.tree.data[idx:]  + self.tree.data[: idx + (multi_step - 1) - self.max_size]
                else:
                    index_range = self.tree.data[idx : min(idx + multi_step, self.tree.current_size)]   
                
                # unpackage the rewards and dones
                n_step_sample = [(index[-2], index[-1]) for index in index_range]                
                rewards, dones = map(list, zip(*n_step_sample))     
                
                # cut the reward to the appropriate length
                try: 
                    done_index = dones.index(True) + 1 
                    done_val = True
                
                except: 
                    done_index = len(rewards) 
                    done_val = False
                
                # udpdate the done_index if it will take data from overlap 
                diff = self.tree.write - (done_index + idx)
                if diff <= 0 and diff > -multi_step:        
                    done_index = done_index + diff - 1                    
                    done_val = dones[done_index - 1]  
                
                # discount the rewards
                rewards = rewards[:done_index]                
                reward_val = sum([reward * (gamma ** idx) for idx, reward in enumerate(rewards)])
                
                # get the index for the next state before termination
                next_state_indexes.append((done_index + idx) % self.max_size)
                reward.append(reward_val)
                done.append(done_val) 
                
            # get the 0th state for the batch       
            state, action, _, _, _ = map(torch.tensor, zip(*batch))
            
            # get the nth state
            next_batch = [self.tree.data[idx] for idx in next_state_indexes]
            _, _, next_state, _, _ = map(torch.tensor, zip(*next_batch))
            
            # convert to tensor
            done = torch.tensor(done)
            reward = torch.tensor(reward)
        
        else:
            idxs, priorities, batch, _ = zip(*outputs)              
            
            # convert the batch into an appropriate tensor form        
            state, action, next_state, reward, done = map(torch.tensor, zip(*batch))
                    
        # calculate importance sampling weights        
        # Added a small 1e-8 factor to stop warning with zero priority
        tree_total, tree_size = self.tree.total(), self.tree.current_size         
        is_weights = [tree_size * (((priority + 1e-8) /tree_total) ** (-self.beta)) for _, priority in enumerate(priorities)]
        is_weights_max = max([is_weights])[0]        
        is_weights = [is_weight / is_weights_max for _, is_weight in enumerate(is_weights)]
        
        # run the tensors on the selected device
        state = state.to(device)
        action = action.to(device)
        reward = reward.to(device)
        next_state = next_state.to(device)
        done = done.to(device)
        
        # convert importance sampling weights to tensor form
        is_weights = torch.FloatTensor(is_weights)
        is_weights = is_weights.to(device)
        
        # make all the tensors 2D
        reward = reward.unsqueeze(1)
        done = done.unsqueeze(1)
        is_weights = is_weights.unsqueeze(1)
        
        # check the dimension of the action and convert to 2D
        if len(action.size()) == 1: 
            action = action.unsqueeze(1)
        
        # repackage the batch
        batch_sample = (state, action, next_state, reward, done)

        return batch_sample, idxs, is_weights 
    
    def get_length(self):
        return self.tree.current_size

    def _get_priority(self, error):
        return (error + self.epsilon) ** self.alpha       
    
    def update(self, idx, error):
        
        # get the proprity 
        p = self._get_priority(error)
        
        # update the max_priority
        if p > self.max_priority:
            self.max_priority = p
        
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
        idx = self.write + (self.max_size - 1)
        
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
        self._propagate(idx, change)
        
        
    # update to the root node
    def _propagate(self, idx, change):
        
        # get the parent node
        parent = (idx - 1) // 2
        
        # add sum to parent node
        self.tree[parent] += change
        
        # if this isn't the root node -> recursion
        if parent != 0:
            self._propagate(parent, change)
            
            
    # get priority and sample
    def get(self, sample):
        
        # get the index
        idx = self._retrieve(0, sample)
        
        # get the accompanying data
        dataIdx = idx - self.max_size + 1

        return (idx, self.tree[idx], self.data[dataIdx], dataIdx)
    

    # find sample on leaf node
    def _retrieve(self, idx, sample):
        
        # get the child nodes of the current node
        left = 2 * idx + 1
        right = 2 * idx + 2

        # if this is a leaf node      
        if left >= len(self.tree):
            return idx
        
        # if priority is less than left node follow left node
        if sample <= self.tree[left]:
            return self._retrieve(left, sample)
        
        # follow the right node
        else:
            return self._retrieve(right, sample - self.tree[left])
        
    
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
        
        buffer.push(state=state, action=action, next_state=state, reward=reward, done=done)
        
        if i > 100_000:
            pass
            # print(buffer.buffer[-1])
        
    toc = time.perf_counter()
    print('Appending took {} seconds'.format(toc - tic))    
    print('Final buffer length: {}'.format(buffer.get_length()))  
    
    # test the sampling from the array
    tic_1 = time.perf_counter()
    
    for i in range(10_000):
        
        batch, idxs, is_weight = buffer.sample(batch_size=32, multi_step=1)
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