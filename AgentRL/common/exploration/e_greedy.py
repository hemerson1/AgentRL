#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:28:50 2021

@author: hemerson
"""

""" 
epsilon_greedy - Implementation of an epsilon greedy policy in which the agent
                 has a probability of selecting a random action as opposed to 
                 selecting the action which maximises expected reward

"""

from AgentRL.common.exploration.base import base_exploration

import numpy as np
import torch

# For testing:
from AgentRL.common.value_networks.standard_value_net import standard_value_network

class epsilon_greedy(base_exploration):
    
    def __init__(self, 
                 action_num = 1,
                 device = 'cpu',
                 starting_expl_threshold = 1.0,
                 expl_decay_factor = 0.999, 
                 min_expl_threshold = 0.01
                 ):
         
        self.action_num = action_num
        
        self.device = device
        
        # define the exploration parameters
        self.starting_expl_threshold = starting_expl_threshold
        self.current_exploration = starting_expl_threshold
        self.expl_decay_factor = expl_decay_factor
        self.min_expl_threshold = min_expl_threshold        
        
    def get_action(self, value_network, state):
                
        # get a random number
        random_num = np.random.uniform(0,1,1)[0]
        
        # get a random action
        if random_num <= self.current_exploration:
            action = np.random.randint(0, self.action_num, size=(1,))
            
        # take the action(s) with the max Q value(s)
        else:
            
            # convert to tensor and select argmax
            tensor_state = torch.FloatTensor(state).to(self.device)
            with torch.no_grad():
                action = torch.argmax(value_network(tensor_state)).unsqueeze(0)
                action = action.cpu().data.numpy()
                
        return action
    
    def reset(self):
        
        # reset the current exploration to the starting
        self.current_exploration = self.starting_expl_threshold
    
    def update(self):
        
        # multiply by decay factor
        self.current_exploration *= self.expl_decay_factor
        
        # check value does not exceed minimum
        self.current_exploration = max(self.current_exploration, 
                                       self.min_expl_threshold)
        
# TESTING ###################################################
                
if __name__ == "__main__":    
    
    # define a test environment
    state_dim = 2
    action_num = 8
    hidden_dim = 10
    
    # Initialise value net and epsilon greedy policy
    value_net = standard_value_network(state_dim, action_num, hidden_dim=hidden_dim)    
    exp = epsilon_greedy(action_num)
    
    # change from random to argmax policy
    # exp.current_exploration = 0.0
    
    # get a test state
    test_state = np.array([10, 5], dtype=np.int32)
    
    # get the e-greedy action
    exp.current_exploration = 0
    action = exp.get_action(value_net, test_state)
    
    # show the action's value, shape and dim
    # print(action)
    # print(action.shape)
    # print(type(action))
    
    # Test the update function
    for i in range(1, 10_000 + 1):
        
        if i % 1_000 == 0:
            print('Episode {} - Exploration {}'.format(i, exp.current_exploration))
        
        exp.update()
    
    # Test the reset function
    exp.reset()
    print('Final exploration {}'.format(exp.current_exploration))
    
#################################################################
        
        
        
    
    
    
    
    