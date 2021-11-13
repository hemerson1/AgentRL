#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:01:19 2021

@author: hemerson
"""

from AgentRL.agents.base import base_agent

class DQN(base_agent):
    
    # TODO: map out the rough structure of the algorithm
    
    def __init__(self, input_type="array", 
                 replay_buffer = None,
                 exploration_strategy="greedy",
                 batch_size = 64,
                 gamma = 0.99
                 ):
        
        # TODO: update the default hyperparameters
        
        # Raise implementation errors
        # TODO: add an assertion to check if replay buffer is valid (not just not None)
        
        # Set the parameters of the environment
        self.input_type = input_type
        
        # Set the structure of the agent
        self.replay_buffer = replay_buffer
        self.exploration_strategy = exploration_strategy
        
        # Set the hyperparameters 
        self.batch_size = batch_size
        self.gamma = gamma
    
    def update(self):
        
        # Sample a batch from the replay buffer
        
        # Use the Q network to predict the Q values for the current states
        
        # Use the Q network to predict the Q values for the next states
        
        # Compute the updated Q value using:
        # Q_exp = r + (1 - done) * done * max(next_Q)
        
        # Compute the loss - the MSE of the current and the expected Q value
        
        # Perform a gradient update        
        
        raise NotImplementedError        
        
    def get_action(self, state):
        
        # For epsilon - greedy
        
        # Get the exploration val at this timestep
        
        # If exploration > threshold: take random discrete action
        
        # Else: select the action corresponding to the largest Q_value for that state
        
        # return the action        
        
        raise NotImplementedError        
    
    def save_model(self):
        raise NotImplementedError         
        
    def load_model(self):
        raise NotImplementedError 

if __name__ == '__main__':
    
    agent = DQN()