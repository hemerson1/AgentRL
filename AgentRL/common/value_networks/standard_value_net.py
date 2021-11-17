#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:30:59 2021

@author: hemerson
"""

from AgentRL.common.value_networks.base import base_value_network

import torch.nn as nn
import torch.nn.functional as F

class standard_value_network(base_value_network):
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, activation=F.relu):   
        super().__init__()
        
        # initialise the layers
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, action_dim)
        
        # get the activation function
        self.activation = activation
    
    def forward(self, state):
        
        x = state        
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        x = self.linear_4(x)
        
        return x      
    
# TESTING ###################################################
        
if __name__ == '__main__':
    
    Q_net = standard_value_network(10, 2)
    
#################################################################