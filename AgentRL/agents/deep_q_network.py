#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:01:19 2021

@author: hemerson
"""

from AgentRL.agents.base import base_agent
from AgentRL.common.value_networks.standard_value_net import standard_value_network
from AgentRL.common.exploration.e_greedy import epsilon_greedy

import numpy as np
import torch
import torch.nn.functional as F

# Testing:
from AgentRL.common.buffers.standard_buffer import standard_replay_buffer

# Inspiration for the implementation was taken from:
# https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

class DQN(base_agent):
    
    def __init__(self, 
                 
                 # Environment
                 state_dim,
                 action_num,
                 action_dim = 1,
                 input_type = "array", 
                 
                 # Hyperparameters
                 hidden_dim = 32, 
                 batch_size = 64,
                 gamma = 0.99,
                 learning_rate = 1e-3,
                 
                 # Update
                 update_method = "hard",
                 update_interval = 1
                 tau = 1e-2,
                 
                 # Replay 
                 replay_buffer = None,
                 
                 # Exploration
                 exploration_strategy = "greedy",
                 starting_expl_threshold = 1.0,
                 expl_decay_factor = 0.999, 
                 min_expl_threshold = 0.01
                 ):
        
        # TODO: add cpu and gpu compatibility
        
        # TODO: how will you make sure the string options selected are correct?
        
        # TODO: update the default hyperparameters
        
        # Raise implementation errors
        # TODO: add an assertion to check if replay buffer is valid (not just not None)
        
        # Set the parameters of the environment
        self.state_dim = state_dim
        self.action_num = action_num
        self.action_dim = action_dim
        self.input_type = input_type
        
        # TODO: set the torch, numpy and random seed
        
        # Set the hyperparameters 
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        # Set the structure of the agent
        self.replay_buffer = replay_buffer
        self.q_net = standard_value_network(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim)    
        self.target_q_net = standard_value_network(self.state_dim, self.action_dim, hidden_dim=self.hidden_dim)    
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        
        # Configure the exploration strategy
        
        # set-up the e - greedy policy
        self.exploration_strategy = exploration_strategy   
        
        if exploration_strategy == "greedy":
            self.policy = epsilon_greedy(
                self.action_num,
                starting_expl_threshold = starting_expl_threshold,
                expl_decay_factor = expl_decay_factor, 
                min_expl_threshold = min_expl_threshold                                        
            )
            
                
    def update(self):
        
        # Sample a batch from the replay buffer
        if self.replay_buffer.get_length() > self.batch_size:
            state, action, reward, next_state, done  = self.replay_buffer.sample(batch_size=self.batch_size)
        else:
            return
        
        # TODO: make sure that the replay outputs tensors
        
        # TODO: finish update implementation
        
        # Use the Q network to predict the Q values for the current states
        current_Q = self.q_net(state)
        
        # Use the Q network to predict the Q values for the next states
        next_Q = self.target_q_net(next_state)
        
        # Compute the updated Q value using:
        target_Q = reward + (1 - done) * self.gamma * torch.max(next_Q)
        
        # Compute the loss - the MSE of the current and the expected Q value
        loss = F.smooth_l1_loss(current_Q, target_Q)
        
        # Perform a gradient update        
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        
        # TODO: Add internal counter for timesteps

        # Perform a hard update     
        if self.update_method == 'hard':
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        
        # Perform a soft update 
        elif self.update_method == 'soft':
            for target_param, orig_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(self.tau * orig_param.data + (1.0 - self.tau) * target_param.data)
                
        
    def get_action(self, state): 
        
        # For epsilon - greedy
        if self.exploration_strategy == "greedy":             
            action = self.policy.get_action(self.q_net, state)
            
            # update the exploration params
            self.policy.update()
            
            return action     
    
    def save_model(self):
        raise NotImplementedError         
        
    def load_model(self):
        raise NotImplementedError 

if __name__ == '__main__':
    
    # Set up the test params
    state_dim = 2
    action_num = 9
    state = np.array([10, 2], dtype=np.float32)
    reward = 2
    done = False
    replay_size = 5_000
    
    # Intialise the buffer
    buffer = standard_replay_buffer(max_size=replay_size)
    
    # Initialise the agent
    agent = DQN(state_dim=2, 
                action_num=9,
                replay_buffer=buffer) 
    
    # Create an update loop 
    print('Starting exploration: {}'.format(agent.policy.current_exploration))
    for timestep in range(1, 10_000 + 1):        

        # get an agent action
        action = agent.get_action(state)
        
        # push test samples to the replay buffer
        buffer.push(state=state, action=action, 
                    next_state=state, reward=reward, done=done)
        
        # display the test parameters
        if timestep % 1000 == 0:
            print('Steps {}'.format(timestep))
            print('------------------------------')
            print('Current buffer length {}'.format(buffer.get_length()))
            print('Current action: {}/{}'.format(action[0], action_num - 1))
            print('Exploration: {}'.format(agent.policy.current_exploration))
            print('------------------------------')
        
        # update the agent's policy
        agent.update()
    
    print('Selected action: {}/{}'.format(action[0], action_num - 1))
    
    
    
    