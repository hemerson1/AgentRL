#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:33:39 2021

@author: hemerson
"""

class base_exploration():
    
    def __init__(self):
        
        # TODO: edit this inheritance to add syntax
        
        """ 
        Each value network must have the following variables: 
            
        self.buffer_size - int (how many samples does the buffer store?)
        
        """      
        pass    
    
    def get_action(self, q_network):
        raise NotImplementedError
    
    