#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 12:01:19 2021

@author: hemerson
"""

""" 
DQN - An implementation of a deep Q network originally introduced in:
      https://arxiv.org/abs/1312.5602
      
Includes the following modifications:
    - double
    - duelling
    - compatibility with prioritised replay
    - noisy neural network learning
    - categorical
    - multi-step
    - rainbow

"""

from AgentRL.agents.base import base_agent
from AgentRL.common.value_networks import standard_value_network, duelling_value_network
from AgentRL.common.exploration import epsilon_greedy, default_argmax

import numpy as np
import random
import torch
import torch.nn.functional as F

# TESTING:
from AgentRL.common.buffers import standard_replay_buffer, prioritised_replay_buffer

import time

# Inspiration for basic DQN structure was taken from:
# https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

# Guidance for categorical DQN structure was taken from:
# https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/06.categorical_dqn.ipynb

# Rainbow
# https://github.com/maybe-hello-world/RainbowDQN/tree/newhope/Rainbow


class DQN(base_agent):
    
    # TODO: add compatibility for input_type
    # TODO: hard to know distribution of V_min and V_max -> this can be solved with quantile rergression (Possibly the next step)
    # TODO: tidy up all the N step code
    # TODO: make sure the action_num is compatible with the numpy array format of the SimpleRL environments
    
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
                 min_expl_threshold = 0.01,
                 
                 # Categorical 
                 categorical = False,
                 v_range = (0, 100),
                 atom_size = 50,
                 
                 # Multi-step
                 multi_step = 1
                 
                 ):        
        
        # Raise implementation errors
        
        # Ensure the target update method is valid for DQN
        valid_target_update_methods = ["soft", "hard"]
        target_update_method_error = "target_update_method is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_target_update_methods)
        assert target_update_method in valid_target_update_methods, target_update_method_error
        
        # Ensure the algorithm type method is valid for DQN
        valid_algorithm_methods = ["default", "double", "duelling", "rainbow"]
        algorithm_method_error = "algorithm_type is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_algorithm_methods)
        assert algorithm_type in valid_algorithm_methods, algorithm_method_error
        
        # Ensure the Exploration method is valid for DQN
        valid_exploration_methods = ["greedy", "noisy_network"]
        exploration_method_error = "exploration_method is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_exploration_methods)
        assert exploration_method in valid_exploration_methods, exploration_method_error
        
        # Ensure the replay type is valid for DQN
        valid_replays = ["default", "prioritised"]
        replay_error = "replay is not valid for this agent, " \
            + "please select one of the following: {}.".format(valid_replays)
        assert replay_buffer.buffer_name in valid_replays, replay_error
        
        # Ensure the action_dim is correct
        action_dim_error = "action_dim is invalid, the current implementation only " \
            + "supports 1D action spaces."
        assert action_dim == 1, action_dim_error
        
        # Display the input settings 
        print('--------------------')
        print('DQN SETTINGS:')
        print('--------------------')
        print('State dim: {}'.format(state_dim))    
        print('Action dim: {}'.format(action_dim))  
        print('Input type: {}'.format(input_type))      
        print('Seed: {}'.format(seed))
        print('Device: {}'.format(device))        
        print('\nHyperparameters:')
        print('--------------------')
        print('Algorithm type: {}'.format(algorithm_type))
        print('Hidden dimensions: {}'.format(hidden_dim))
        print('Batch size: {}'.format(batch_size))
        print('Discount factor: {}'.format(gamma))
        print('Learning rate: {}'.format(learning_rate))
        print('Steps per network update: {}'.format(network_update_freq))
        print('Target Update Method: {}'.format(target_update_method))
        if target_update_method == "hard":
            print('Steps per target update: {}'.format(target_update_freq))        
        if target_update_method == "soft":
            print('Soft target update factor: {}'.format(tau))
        print('Replay type: {}'.format(replay_buffer.buffer_name))
        if algorithm_type == "rainbow":
            print('Exploration Method: noisy_network')
        else:
            print('Exploration Method: {}'.format(exploration_method))
        if exploration_method == "greedy" and algorithm_type != "rainbow":
            print('Starting exploration: {}'.format(starting_expl_threshold))
            print('Exploration decay factor: {}'.format(expl_decay_factor))
            print('Minimum exploration: {}'.format(min_expl_threshold))
        print('Categorical Learning: {}'.format(categorical))
        if categorical or algorithm_type == "rainbow":
            print('Value function range: {}'.format(v_range))
            print('Categorical atom size: {}'.format(atom_size))
        if multi_step > 1:
            print('Multi-step learning: {}'.format(multi_step))
        print('--------------------\n')
                
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
        
        # Set noisy network parameters
        self.noisy = False
        
        # Configure categorical structure
        self.categorical = categorical
        self.v_min, self.v_max = v_range
        self.atom_size = atom_size
        self.delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1) 
        self.support = None
        
        # Set Multi-step parameters
        self.multi_step = multi_step
        
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
        if self.exploration_method == "greedy" and self.algorithm_type != "rainbow": 
            
            # load in e-greedy policy
            self.policy = epsilon_greedy(
                self.action_num,
                self.device,
                starting_expl_threshold = self.starting_expl_threshold,
                expl_decay_factor = self.expl_decay_factor, 
                min_expl_threshold = self.min_expl_threshold                                        
            )  
            
        elif self.exploration_method == "noisy_network" or self.algorithm_type == "rainbow":
            
            # load in argmax policy
            self.policy = default_argmax(
                action_num=self.action_num,
                device=self.device                                     
            )  
            
            # set network structure to noisy
            self.noisy = True        
            
        # Reset the support for categorical DQN
        if self.categorical or self.algorithm_type == "rainbow":
            self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)
            
        # Empty the replay buffer
        self.replay_buffer.reset()
        
        # Initialise the default DQN network
        if self.algorithm_type == "default":
            self.q_net = standard_value_network(self.state_dim, self.action_num, hidden_dim=self.hidden_dim, noisy=self.noisy,
                                                categorical=self.categorical, v_min=self.v_min, v_max=self.v_max,
                                                atom_size=self.atom_size, support=self.support, device=self.device).to(self.device) 
            
        # Initialise the double DQN networks
        elif self.algorithm_type == "double":
            self.q_net = standard_value_network(self.state_dim, self.action_num, hidden_dim=self.hidden_dim, noisy=self.noisy,
                                                categorical=self.categorical, v_min=self.v_min, v_max=self.v_max, 
                                                atom_size=self.atom_size, support=self.support, device=self.device).to(self.device) 
            self.target_q_net = standard_value_network(self.state_dim, self.action_num,hidden_dim=self.hidden_dim, noisy=self.noisy,
                                                       categorical=self.categorical, v_min=self.v_min, v_max=self.v_max,
                                                       atom_size=self.atom_size, support=self.support, device=self.device).to(self.device) 
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        # Initialise the duelling DQN networks
        elif self.algorithm_type == "duelling" or self.algorithm_type == "rainbow":
            self.q_net = duelling_value_network(self.state_dim, self.action_num, hidden_dim=self.hidden_dim, noisy=self.noisy,
                                                categorical=self.categorical, v_min=self.v_min, v_max=self.v_max,
                                                atom_size=self.atom_size, support=self.support, device=self.device).to(self.device) 
            self.target_q_net = duelling_value_network(self.state_dim, self.action_num, hidden_dim=self.hidden_dim, noisy=self.noisy,
                                                       categorical=self.categorical, v_min=self.v_min, v_max=self.v_max,
                                                       atom_size=self.atom_size, support=self.support, device=self.device).to(self.device) 
            self.target_q_net.load_state_dict(self.q_net.state_dict())
            
        # Initialise the optimiser
        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)          
                
    def update(self):
        
        # Update the network at the specified interval
        if self.current_step % self.network_update_freq == 0:
        
            # Sample a batch from the replay buffer
            if self.replay_buffer.get_length() > self.batch_size + (self.multi_step - 1):
                
                if self.replay_buffer_name == "default":
                    state, action, next_state, reward, done  = self.replay_buffer.sample(batch_size=self.batch_size,
                                                                                         device=self.device, multi_step=self.multi_step, gamma=self.gamma)
                    
                elif self.replay_buffer_name == "prioritised":
                    
                    # get a batch as well as idxs and importance sampling weights                    
                    batch, idxs, is_weights = self.replay_buffer.sample(batch_size=self.batch_size, device=self.device,
                                                                        multi_step=self.multi_step, gamma=self.gamma)
                    state, action, next_state, reward, done = batch      
                    
            else:
                return
            
            # ensure that the state and next state are floats
            # this is required to be input into the feedforward neural network
            state = state.type(torch.float32)
            next_state = next_state.type(torch.float32)
            
            if self.categorical or self.algorithm_type == "rainbow":
                
                with torch.no_grad():
                    
                    # Use the Q network to predict the next action and the distribution of returns
                    if self.algorithm_type == 'default':                        
                        next_action = self.q_net(next_state).argmax(1)
                        next_distribution = self.q_net(next_state, get_distribution=True) 
                        
                    # Use the target Q network to predict the  next action and distribution of returns for the next states    
                    elif self.algorithm_type == "double" or self.algorithm_type == "duelling" or self.algorithm_type == "rainbow": 
                        next_action = self.target_q_net(next_state).argmax(1)                        
                        next_distribution = self.target_q_net(next_state, get_distribution=True)                         
                        
                    next_distribution = next_distribution[range(self.batch_size), next_action]
                    
                    # perform a bellman update of the distribution of returns
                    not_done = ~done
                    z = (reward + not_done * (self.gamma ** self.multi_step) * self.support).clamp(min=self.v_min, max=self.v_max)
                    
                    # calculate the atom of each return and upper and lower bound
                    b = (z - self.v_min) / self.delta_z
                    l = b.floor().long()
                    u = b.ceil().long()
        
                    # get the true atom structure
                    offset = (
                        torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size
                        ).long()
                        .unsqueeze(1)
                        .expand(self.batch_size, self.atom_size)
                        .to(self.device)
                    )
        
                    # determine the projected atom structure using the offset and predicted distribution
                    proj_distribution = torch.zeros(next_distribution.size(), device=self.device)                    
                    proj_distribution.view(-1).index_add_(
                        0, (l + offset).view(-1), (next_distribution * (u.float() - b)).view(-1)
                    )                    
                    proj_distribution.view(-1).index_add_(
                        0, (u + offset).view(-1), (next_distribution * (b - l.float())).view(-1)
                    )
                    
                # get the distribution of log returns
                distribution = self.q_net(state, get_distribution=True)  
                
                # Added squeeze to produce log_p with correct dimensions
                log_p = torch.log(distribution[range(self.batch_size), action.squeeze(1)])
                
                # Compute the loss
                if self.replay_buffer_name == "default":                    
                    loss = -(proj_distribution * log_p).sum(1).mean()                    
                
                # update the priority for the batch and compute the loss
                elif self.replay_buffer_name == "prioritised":

                    # calculate the elementwise loss and factor in importance weights
                    elementwise_loss = -(proj_distribution * log_p).sum(1)                    
                    loss = (elementwise_loss * is_weights).mean()
                    
                    # update the errors for the batch
                    errors = torch.abs(elementwise_loss).cpu().data.numpy()
                                    
                    for i, error in enumerate(errors):
                        self.replay_buffer.update(idxs[i], error)  
                    
            else:            
            
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
                target_Q = reward + not_done * (self.gamma ** self.multi_step) * torch.max(next_Q, dim=1, keepdim=True).values
                
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
            if self.algorithm_type == "double" or self.algorithm_type == "duelling" or self.algorithm_type == "rainbow":
            
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
        
        # For epsilon - greedy
        if self.exploration_method == "greedy" and self.algorithm_type != "rainbow":         
            action = self.policy.get_action(self.q_net, state)
            
            # update the exploration params
            self.policy.update()
            self.current_exploration = self.policy.current_exploration
            
        # For noisy neural network exploration
        elif self.exploration_method == "noisy_network" or self.algorithm_type == "rainbow":
            action = self.policy.get_action(self.q_net, state)   
            
        return action          
        
    def push(self, state, action, next_state, reward, done):        
        self.replay_buffer.push(state=state, action=action, next_state=next_state, reward=reward, done=done)
            

    def save_model(self, path):
        
        # save the q network
        torch.save(self.q_net.state_dict(), path + '_q_network')
        
        # save the target network
        if self.algorithm_type == "double" or self.algorithm_type == "duelling" or self.algorithm_type == "rainbow":
            torch.save(self.target_q_net.state_dict(), path + '_target_q_network')
        
    def load_model(self, path):
        
        # load the q network
        self.q_net.load_state_dict(torch.load(path +'_q_network'))
        self.q_net.eval()
        
        # load the target network
        if self.algorithm_type == "double" or self.algorithm_type == "duelling"  or self.algorithm_type == "rainbow":
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
    replay_size = 10_000
    
    # Intialise the buffer
    # buffer = None # A non existent buffer
    # buffer = base_buffer() # buffer with unimplemented features
    # buffer = standard_replay_buffer(max_size=replay_size)
    buffer = prioritised_replay_buffer(max_size=replay_size, seed=0)
    
    # Initialise the agent
    agent = DQN(state_dim=state_dim, 
                action_num=action_num,
                action_dim=action_dim,
                replay_buffer=buffer,
                target_update_method="hard", 
                exploration_method="noisy_network",
                algorithm_type="default",
                categorical=False, 
                seed=0
                ) 
    
    # Create an update loop 
    if agent.exploration_method == 'greedy':  
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
            if agent.exploration_method == 'greedy':            
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
    if agent.exploration_method == 'greedy':  
        print('Reset Exploration: {}'.format(agent.current_exploration))
    print('Completed in {} s'.format(toc - orig_tic)) 
    print('------------------------------')   
    
#################################################################
    
    
    
    
    
    
    
    
    