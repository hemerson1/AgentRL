#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:15:01 2021

@author: hemerson
"""

class base_buffer():
    
    def __init__(self):
        
        # TODO: fill in the structure of the inheritance class
        """ 
        Each agent must have the following variables: 
            
        self.buffer_size - int (how many samples does the buffer store?)
        
        """                
        pass
    
    def reset(self):
        raise NotImplementedError

    def push(self):
        raise NotImplementedError        
        
    def sample(self):
        raise NotImplementedError
        
    def get_length(self):
        raise NotImplementedError
        
        
if __name__ == '__main__':
    
    pass
