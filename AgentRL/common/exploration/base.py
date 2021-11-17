#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 17:33:39 2021

@author: hemerson
"""

""" 
base_exploration - A template for implementing agent policies

"""

class base_exploration():
    
    def __init__(self, action_num):        
        """ 
        No variables are required by the component
        
        """     
        pass  
    
    def reset(self):
        """ 
        Resets the parameters of the exploration policy
        
        Inputs:
        --------
        None
        
        Returns:
        ----------
        None 
        """
        raise NotImplementedError
    
    def get_action(self, network, state):
        """ 
        Resets the parameters of the exploration policy
        
        # TODO: check what the correct input type is for the neural net
        
        Inputs:
        --------
        network - torch_neural_network (the neural network used to determine 
                                        the policy )
        state - np.float32 (The current state of the environment)
        
        Returns:
        ----------
        None 
        """        
        raise NotImplementedError
    
    