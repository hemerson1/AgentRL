#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:30:59 2021

@author: hemerson
"""

""" 
standard_value_network -  a simple feedforward value network

"""

from AgentRL.common.value_networks.base import base_value_network
from AgentRL.common.layers.custom_layers import factorised_noisy_linear_layer

import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspiration for duelling_value_network
# https://github.com/gouxiangchen/dueling-DQN-pytorch/blob/master/dueling_dqn.py

class standard_value_network(base_value_network):
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, activation=F.relu, noisy=False):   
        super().__init__()
        
        # initialise the layers
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        
        # specify the linear layer type
        if noisy: linear_layer = factorised_noisy_linear_layer
        else: linear_layer = nn.Linear
        
        # add the linear layers
        self.linear_2 = linear_layer(hidden_dim, hidden_dim)
        self.linear_3 = linear_layer(hidden_dim, hidden_dim)
        self.linear_4 = linear_layer(hidden_dim, action_dim)    
        
        # get the activation function
        self.activation = activation
    
    def forward(self, state):
        
        x = state        
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.linear_4(x)
        
        return x      
    
class duelling_value_network(base_value_network):
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, activation=F.relu, noisy=False):   
        super().__init__()
        
        # initialise the layers
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        
        # specify the linear layer type
        if noisy: linear_layer = factorised_noisy_linear_layer
        else: linear_layer = nn.Linear
        
        # add the linear layer
        self.linear_2 = linear_layer(hidden_dim, hidden_dim)
            
        # Value function layers
        self.value_1 = linear_layer(hidden_dim, hidden_dim)
        self.value_2 = linear_layer(hidden_dim, 1)
        
        # Advantage function layers
        self.advantage_1 = linear_layer(hidden_dim, hidden_dim)
        self.advantage_2 = linear_layer(hidden_dim, action_dim)        
        
        # get the activation function
        self.activation = activation
    
    def forward(self, state):
        
        x = state        
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        
        # Value function
        value = self.activation(self.value_1(x))
        value = self.value_2(value)
        
        # Advantage function
        advantage = self.activation(self.advantage_1(x))
        advantage = self.advantage_2(advantage)
        
        mean_advantage = torch.mean(advantage, dim=1, keepdim=True)
        Q = value + advantage - mean_advantage
        
        return Q 
    
    
# TESTING ###################################################
        
if __name__ == '__main__':
    
    Q_net = standard_value_network(10, 2)
    
#################################################################