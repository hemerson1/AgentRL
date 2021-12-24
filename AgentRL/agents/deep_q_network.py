#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:01:19 2021

@author: hemerson
"""

""" 
DQN - An implementation of a deep Q network originally introduced in:
      https://arxiv.org/abs/1312.5602

"""

from AgentRL.agents.base import base_agent
from AgentRL.common.value_networks.standard_value_net import standard_value_network, duelling_value_network
from AgentRL.common.exploration.e_greedy import epsilon_greedy
from AgentRL.common.buffers.base import base_buffer

import numpy as np
import random
import torch
import torch.nn.functional as F

# TESTING:
from AgentRL.common.buffers.standard_buffer import standard_replay_buffer
from AgentRL.common.buffers.prioritised_buffer import prioritised_replay_buffer

import time

# Inspiration for the implementation was taken from:
# https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

class DQN(base_agent):
    
    # TODO: add compatibility for input_type
    # TODO: print the hyperparameters on initialise
    # TODO: add the following DQN variations: Double (DONE), Duelling (DONE), Prioritised (DONE), Noisy, Categorical, Rainbow
    # TODO: should they be able to implement a combination? e.g. Double and Duelling
    
    def __init__(self, 
                 
                 # Environment
                 state_dim,
                 action_num,
                 action_dim = 1,
                 input_type = "array", 
                 seed = None,
                 
                 # Device
                 device = 'cpu',
                 
                 # Hyperparameters
                 algorithm_type='default',
                 hidden_dim = 32, 
                 batch_size = 32,
                 gamma = 0.99,
                 learning_rate = 1e-3,
                 noisy = False,
                 
                 # Update
                 network_update_freq = 1,
                 target_update_method = "hard",
                 target_update_freq = 10,
                 tau = 1e-2,
                 
                 # Replay 
                 replay_buffer = None,
                 
                 # Exploration
                 exploration_method = "greedy",
                 starting_expl_threshold = 1.0,
                 expl_decay_factor = 0.999, 
                 min_expl_threshold = 0.01
                 
                 ):        
        
        # Raise implementation errors
        
        # Ensure the target update method is valid for DQN
        valid_target_update_methods = ["soft", "hard"]
        target_update_method_error = "target_update_method is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_target_update_methods)
        assert target_update_method in valid_target_update_methods, target_update_method_error
        
        # Ensure the algorithm type method is valid for DQN
        valid_algorithm_methods = ["default", "double", "duelling"]
        algorithm_method_error = "algorithm_type is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_algorithm_methods)
        assert algorithm_type in valid_algorithm_methods, algorithm_method_error
        
        # Ensure the Exploration method is valid for DQN
        valid_exploration_methods = ["greedy"]
        exploration_method_error = "exploration_method is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_exploration_methods)
        assert exploration_method in valid_exploration_methods, exploration_method_error
        
        # Ensure the replay buffer is valid
        test_buffer =  base_buffer
        replay_buffer_error = "replay_buffer is invalid, please ensure a buffer from " \
            + "'AgentRL/common/buffers' is utilised. If you are trying to implement a custom " \
            + "buffer ensure that it is built inline with the template in base.py." 
        assert isinstance(replay_buffer, test_buffer), replay_buffer_error
        
        # Ensure the action_dim is correct
        action_dim_error = "action_dim is invalid, the current implementation only " \
            + "supports 1D action spaces."
        assert action_dim == 1, action_dim_error
        
        # Set the parameters of the environment
        self.state_dim = state_dim
        self.action_num = action_num
        self.action_dim = action_dim
        self.input_type = input_type
        
        # Set the torch, numpy and random seed
        self.seed = seed     
        
        # Set the device
        self.device = device
        
        # Set the hyperparameters 
        self.algorithm_type = algorithm_type
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.tau = tau
        
        # Set the replay buffer
        self.replay_buffer = replay_buffer
        self.replay_buffer_name = replay_buffer.buffer_name
        
        # Set the update frequency
        self.target_update_method = target_update_method
        self.network_update_freq = network_update_freq
        self.target_update_freq = target_update_freq
        
        # Configure the exploration strategy
        self.exploration_method = exploration_method   
        
        # Set the e - greedy parameters
        self.starting_expl_threshold = starting_expl_threshold
        self.expl_decay_factor = expl_decay_factor
        self.min_expl_threshold = min_expl_threshold
        self.current_exploration = starting_expl_threshold
        
        # configure the neural network
        self.noisy = noisy
        
        # Reset the policy, network and buffer components
        self.reset()
        
    def reset(self):
        
        # Reset the torch, numpy and random seed
        if type(self.seed) == int:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)  
        
        # Reset the step count for network updates
        self.current_step = 0
        self.network_updates = 0 
        
        # Reset the exploration
        if self.exploration_method == "greedy": 
            self.policy = epsilon_greedy(
                self.action_num,
                self.device,
                starting_expl_threshold = self.starting_expl_threshold,
                expl_decay_factor = self.expl_decay_factor, 
                min_expl_threshold = self.min_expl_threshold                                        
            )      
            
        # Empty the replay buffer
        self.replay_buffer.reset()
        
        # Initialise the default DQN network
        if self.algorithm_type == "default":
            self.q_net = standard_value_network(self.state_dim, self.action_num,
                                                hidden_dim=self.hidden_dim, noisy=self.noisy).to(self.device) 
            
        # Initialise the double DQN networks
        elif self.algorithm_type == "double":
            self.q_net = standard_value_network(self.state_dim, self.action_num,
                                                hidden_dim=self.hidden_dim, noisy=self.noisy).to(self.device) 
            self.target_q_net = standard_value_network(self.state_dim, self.action_num,
                                                       hidden_dim=self.hidden_dim, noisy=self.noisy).to(self.device) 
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        # Initialise the duelling DQN networks
        elif self.algorithm_type == "duelling":
            self.q_net = duelling_value_network(self.state_dim, self.action_num,
                                                hidden_dim=self.hidden_dim, noisy=self.noisy).to(self.device) 
            self.target_q_net = duelling_value_network(self.state_dim, self.action_num,
                                                       hidden_dim=self.hidden_dim, noisy=self.noisy).to(self.device) 
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        # Initialise the optimiser
        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)          
                
    def update(self):
        
        # Update the network at the specified interval
        if self.current_step % self.network_update_freq == 0:
        
            # Sample a batch from the replay buffer
            if self.replay_buffer.get_length() > self.batch_size:
                
                if self.replay_buffer_name == "default":
                    state, action, next_state, reward, done  = self.replay_buffer.sample(batch_size=self.batch_size, device=self.device)
                    
                elif self.replay_buffer_name == "prioritised":
                    
                    # get a batch as well as idxs and importance sampling weights                    
                    batch, idxs, is_weights = self.replay_buffer.sample(batch_size=32)
                    state, action, next_state, reward, done = batch      
                    
            else:
                return
            
            # ensure that the state and next state are floats
            # this is required to be input into the feedforward neural network
            state = state.type(torch.float32)
            next_state = next_state.type(torch.float32)
            
            # Use the Q network to predict the Q values for the current states 
            # and take the Q value for the action that occured
            current_Q = self.q_net(state).gather(1, action)
            
            # Use the Q network to predict the Q values for the next states            
            if self.algorithm_type == "default":
                next_Q = self.q_net(next_state)
            
            # Use the target Q network to predict the Q values for the next states    
            elif self.algorithm_type == "double" or self.algorithm_type == "duelling":
                next_Q = self.target_q_net(next_state)
            
            # Compute the updated Q value using:
            not_done = ~done            
            target_Q = reward + not_done * self.gamma * torch.max(next_Q, dim=1, keepdim=True).values
            
            # Compute the loss - the MSE of the current and the expected Q value
            if self.replay_buffer_name == "default":
                loss = F.smooth_l1_loss(current_Q, target_Q)
            
            # update the priority for the batch and compute the loss
            elif self.replay_buffer_name == "prioritised":
                
                # calculate the errors for the batch
                errors = torch.abs(current_Q - target_Q).cpu().data.numpy()
                                
                for i, error in enumerate(errors):
                    self.replay_buffer.update(idxs[i], error)  
                                
                loss = (is_weights * F.smooth_l1_loss(current_Q, target_Q)).mean()
            
            # Perform a gradient update        
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            
            # update the target network
            if self.algorithm_type == "double" or self.algorithm_type == "duelling":
            
                # Perform a hard update 
                if self.target_update_method == 'hard':
                
                    # Update the target at the specified interval
                    if self.network_updates % self.target_update_freq == 0:
                            self.target_q_net.load_state_dict(self.q_net.state_dict())
                    
                # Perform a soft update 
                elif self.target_update_method == 'soft':
                    for target_param, orig_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                        target_param.data.copy_(self.tau * orig_param.data + (1.0 - self.tau) * target_param.data)
            
            # Update the network update count
            self.network_updates += 1
        
        # Update the timesteps
        self.current_step += 1
                            
    def get_action(self, state): 
        
        #  ensure that the batch dim is set to 1
        if state.ndim < 2:
            state = state.reshape(1, -1)
        
        # For noisy neural network exploration
        if self.noisy:
            action = self.policy.get_action(self.q_net, state)                   
        
        # For epsilon - greedy
        elif self.exploration_method == "greedy":         
            action = self.policy.get_action(self.q_net, state)
            
            # update the exploration params
            self.policy.update()
            self.current_exploration = self.policy.current_exploration
            
        return action          
        
    def push(self, state, action, next_state, reward, done):
        
        if self.replay_buffer_name == "default":
            
            # update the buffer          
            self.replay_buffer.push(state=state, action=action, next_state=next_state, reward=reward, done=done)
            
        elif self.replay_buffer_name == "prioritised":
            
            # TODO: converting to tensors twice seems like an oversight?
            
            # convert to tensors and adjust dimensions
            state_tensor = torch.FloatTensor(state).to(self.device)[None, :]
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)[None, :]
            action_tensor = torch.tensor(action).to(self.device)[None, :]
            reward_tensor = torch.tensor(reward).to(self.device)
            done_tensor = torch.tensor(done).to(self.device)
            
            # Use the Q network to predict the Q values for the current states 
            # and take the Q value for the action that occured
            current_Q = self.q_net(state_tensor).gather(1, action_tensor)
            
            if self.algorithm_type == "default":
                next_Q = self.q_net(next_state_tensor)
            
            # Use the target Q network to predict the Q values for the next states    
            elif self.algorithm_type == "double" or self.algorithm_type == "duelling":
                next_Q = self.target_q_net(next_state_tensor)  
                           
            # Compute the updated Q value using:
            not_done = ~done_tensor            
            target_Q = reward_tensor + not_done * self.gamma * torch.max(next_Q, dim=1, keepdim=True).values
            
            # calculate the error and convert to a 1D numpy array
            error = torch.abs(current_Q - target_Q).cpu().data.numpy().squeeze(0)
            
            # update the buffer with the original numpy sample
            self.replay_buffer.push(error=error, state=state, action=action, next_state=next_state, reward=reward, done=done)
            

    def save_model(self, path):
        
        # save the q network
        torch.save(self.q_net.state_dict(), path + '_q_network')
        
        # save the target network
        if self.algorithm_type == "double" or self.algorithm_type == "duelling":
            torch.save(self.target_q_net.state_dict(), path + '_target_q_network')
        
    def load_model(self, path):
        
        # load the q network
        self.q_net.load_state_dict(torch.load(path +'_q_network'))
        self.q_net.eval()
        
        # load the target network
        if self.algorithm_type == "double" or self.algorithm_type == "duelling":
            self.target_q_net.load_state_dict(torch.load(path +'_target_q_network'))
            self.target_q_net.eval()
        
        
# TESTING ###################################################

if __name__ == "__main__":  

    # Set up the test params
    state_dim = 2
    action_num = 9
    action_dim = 1
    state = np.array([10, 2], dtype=np.int32)
    reward = 2
    done = False
    replay_size = 5_000
    
    # Intialise the buffer
    # buffer = None # A non existent buffer
    # buffer = base_buffer() # buffer with unimplemented features
    # buffer = standard_replay_buffer(max_size=replay_size)
    buffer = prioritised_replay_buffer(max_size=replay_size)
    
    # Initialise the agent
    agent = DQN(state_dim=state_dim, 
                action_num=action_num,
                action_dim=action_dim,
                replay_buffer=buffer,
                target_update_method="hard", 
                exploration_method="greedy",
                algorithm_type="duelling",
                noisy=True
                ) 
    
    # Create an update loop 
    print('Starting exploration: {}'.format(agent.policy.current_exploration))
    
    # test the algorithm speed   
    tic = time.perf_counter()
    orig_tic = tic
    
    for timestep in range(1, 10_000 + 1):        
    
        # get an agent action
        action = agent.get_action(state)
                        
        # push test samples to the replay buffer
        agent.push(state=state, action=action, next_state=state, reward=reward, done=done)

        # display the test parameters
        if timestep % 1000 == 0:
            
            toc = time.perf_counter()
            
            print('\n------------------------------')
            print('Steps {}'.format(timestep))
            print('------------------------------')
            print('Current buffer length {}'.format(buffer.get_length()))
            print('Current action: {}/{}'.format(action[0], action_num - 1))
            print('Exploration: {}'.format(agent.current_exploration))
            print('Completed in {} s'.format(toc - tic)) 
            print('------------------------------')
            
            tic = time.perf_counter()
        
        # update the agent's policy
        agent.update()
    
    print('Selected action: {}/{}'.format(action[0], action_num - 1))
    
    # reset the agent parameters
    agent.reset()
    
    print('\n------------------------------')
    print('Completed')
    print('------------------------------')
    print('Reset buffer length {}'.format(buffer.get_length()))
    print('Reset action: {}/{}'.format(action[0], action_num - 1))
    print('Reset Exploration: {}'.format(agent.current_exploration))
    print('Completed in {} s'.format(toc - orig_tic)) 
    print('------------------------------')   
    
#################################################################
    
    
    
    
    
    
    
    
    