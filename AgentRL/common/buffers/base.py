#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:15:01 2021

@author: hemerson
"""

""" 
base_buffer - A template for implementing experience replays

"""

class base_buffer():
    
    def __init__(self):
        """ 
        No variables are required by the component
        
        """                
        pass
    
    def reset(self):
        """ 
        Resets the experience in the replay buffer 
        
        Inputs:
        --------
        None
        
        Returns:
        ----------
        None        
        """  
        raise NotImplementedError

    def push(self, state, action, reward, next_state, done):
        """ 
        Adds samples to the replay
        
        # TODO: ensure action is input as np.float32 (as adds compatibility 
        with multi-dimensional actions)
        
        Inputs:
        --------
        state - np.float32 (The current state of the environment)
        action - np.float32 (The action taken by the agent)
        reward - float (The reward following taking the current action(s))
        next_state - np.float32 (The next state of the environment)
        done - bool (Is the next state a terminal state?)
        
        Returns:
        ----------
        None        
        """    
        raise NotImplementedError        
        
    def sample(self, batch_size):
        """ 
        Returns a sample from the replay of a specified batch size
        
        Inputs:
        --------
        batch_size - int (The number of samples in the batch)
        
        Returns:
        ----------
        state - np.float32 (The current state of the environment)
        action - np.float32 (The action taken by the agent)
        reward - float (The reward following taking the current action(s))
        next_state - np.float32 (The next state of the environment)
        done - bool (Is the next state a terminal state?)      
        """   
        raise NotImplementedError
        
    def get_length(self):
        """ 
        Returns the current length of the replay buffer
        
        Inputs:
        --------
        None
        
        Returns:
        ----------
        len - int (the current length of the replay buffer)
        """  
        raise NotImplementedError        
        
if __name__ == '__main__':
    
    pass
