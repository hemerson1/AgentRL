#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 17:32:51 2021

@author: hemerson
"""

""" 
epsilon_greedy - Implementation of an epsilon greedy policy in which the agent
                 has a probability of selecting a random action as opposed to 
                 selecting the action which maximises expected reward

"""

from AgentRL.common.exploration import base_exploration

import torch

# For testing:
from AgentRL.common.value_networks import standard_value_network

class default_argmax(base_exploration):
    
    def __init__(self, 
                 action_num = 1,
                 device = 'cpu'
                 ):
         
        self.action_num = action_num        
        self.device = device  
        
    def get_action(self, value_network, state):                
        
        # convert to tensor and select argmax
        tensor_state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = torch.argmax(value_network(tensor_state)).unsqueeze(0).to(self.device)
            action = action.cpu().data.numpy()
                
        return action
    
    def reset(self):
        pass