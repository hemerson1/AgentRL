#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:13:46 2021

@author: hemerson
"""

class base_agent():
    
    def __init__(self):
        
        # TODO: fill in the structure of the inheritance class  
        
        """ 
        Each agent must have the following variables: 
            
        self.input_type - string (what sort of data is being fed to the 
                                  agent? e.g. "array", "image")
        
        """
        
        pass
    
    def update(self):
        raise NotImplementedError        
        
    def get_action(self):
        raise NotImplementedError         
    
    def save_model(self):
        raise NotImplementedError         
        
    def load_model(self):
        raise NotImplementedError 

if __name__ == '__main__':
    
    pass
    