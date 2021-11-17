#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:33:39 2021

@author: hemerson
"""

""" 
base_Q_network - A template for implementing a q network

base_value_network - A template for implementing a value network

"""

from torch import nn

class base_q_network(nn.Module):
    
    def __init__(self):
        super(base_q_network, self).__init__()
        """ 
        No variables are required by the component
        
        """      
        pass
    
    def forward(self, state, action):
        """ 
        Approximates the Q value of a given state and action
        
        Inputs:
        --------
        state - torch.tensorFloat (The current state of the environment)
        action - torch.tensorFloat (The action taken by the agent)
        
        Returns:
        ----------
        Q - torch.tensorFloat (The approximation of the Q value for the state
                               and action)       
        """   
        raise NotImplementedError     
        
class base_value_network(nn.Module):
    
    def __init__(self):
        super(base_value_network, self).__init__()        
        """ 
        No variables are required by the component
        
        """      
        pass
    
    def forward(self, state):
        """ 
        Approximates the value function of a given state 
        
        Inputs:
        --------
        state - torch.tensorFloat (The current state of the environment)
        
        Returns:
        ----------
        value func - torch.tensorFloat (The approximation of the value 
                                        function for the state and action)       
        """   
        raise NotImplementedError    

if __name__ == '__main__':
    pass