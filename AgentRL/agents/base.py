#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 11:13:46 2021

@author: hemerson
"""

""" 
base_agent - A template for the main body of the reinforcement learning agent

"""

class base_agent():
    
    def __init__(self):
                
        """ 
        Each agent must have the following variables: 
            
        state_dim - int (the dimensions of the environment's state space)
        
        Discrete:
        --------
        action_dim - int (the dimensions of the environment's action space)
        action_num - np.int32 (the number of possible actions per action)
        
        Continuous:
        ----------
        # TODO: fill in continuous environment
        
        """        
        pass
    
    
    def reset(self):
        """ 
        Resets the parameters of the learning agent
        
        Inputs:
        --------
        None
        
        Returns:
        ----------
        None        
        """      
        raise NotImplementedError 
        
    
    def update(self):
        """ 
        Updates the network(s) of the learning agent
        
        Inputs:
        --------
        None
        
        Returns:
        ----------
        None        
        """        
        raise NotImplementedError  
        
        
    def get_action(self, state):
        """ 
        Gets an action using the current policy of the learning agent
        
        Inputs:
        --------
        state - np.float32 (the current state of the environment)
        
        Returns:
        ----------
        action - np.int32 (the selected action from the agent)     
        """    
        raise NotImplementedError      
        
    
    def save_model(self, path):
        """ 
        Saves the weights of the learning agent
        
        Inputs:
        --------
        name - str (the path of the weights file to save)
        
        Returns:
        ----------
        None  
        """          
        raise NotImplementedError     
        
        
    def load_model(self, path):
        """ 
        Loads the weights of the learning agent
        
        Inputs:
        --------
        name - str (the path of the weights file to load)
        
        Returns:
        ----------
        None  
        """  
        raise NotImplementedError 
        

if __name__ == '__main__':    
    pass
    