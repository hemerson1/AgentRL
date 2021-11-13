#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:01:19 2021

@author: hemerson
"""

from AgentRL.agents.base import base_agent
from AgentRL.common.value_networks.standard_Q_net import standard_Q_network

# Inspiration for the implementation was taken from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class DQN(base_agent):
    
    # TODO: map out the rough structure of the algorithm
    
    def __init__(self, 
                 
                 # Environment
                 state_dim,
                 action_dim,
                 input_type = "array", 
                 
                 # Hyperparameters
                 hidden_dim = 32, 
                 batch_size = 64,
                 gamma = 0.99,
                 
                 # Replay 
                 replay_buffer = None,
                 
                 # Exploration
                 exploration_strategy = "greedy",
                 starting_expl_threshold = 1.0,
                 expl_decay_factor = 0.999, 
                 min_expl_threshold = 0.01
                 ):
        
        # TODO: update the default hyperparameters
        
        # Raise implementation errors
        # TODO: add an assertion to check if replay buffer is valid (not just not None)
        
        # Set the parameters of the environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_type = input_type
        
        # Set the hyperparameters 
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Set the structure of the agent
        self.replay_buffer = replay_buffer
        self.q_net = standard_Q_network(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim)    
        
        # Configure the exploration strategy
        
        # e - greedy policy
        if exploration_strategy == "greedy":
            
            self.policy = 
            
                
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