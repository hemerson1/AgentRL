#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:33:39 2021

@author: hemerson
"""

from torch import nn

class base_Q_network(nn.Module):
    
    def __init__(self):
        super(base_Q_network, self).__init__()
        
        # TODO: edit this inheritance to add syntax
        
        """ 
        Each value network must have the following variables: 
            
        self.buffer_size - int (how many samples does the buffer store?)
        
        """      
        pass
    
    def forward(self):
        raise NotImplementedError     
        
class base_value_network(nn.Module):
    
    def __init__(self):
        super(base_value_network, self).__init__()
        
        # TODO: edit this inheritance to add syntax
        
        """ 
        Each value network must have the following variables: 
            
        self.buffer_size - int (how many samples does the buffer store?)
        
        """      
        pass
    
    def forward(self):
        raise NotImplementedError    

if __name__ == '__main__':
    pass