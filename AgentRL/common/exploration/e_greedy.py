#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 18:28:50 2021

@author: hemerson
"""

from AgentRL.common.exploration.base import base_exploration

import numpy as np
import torch

class epsilon_greedy(base_exploration):
    
    def __init__(self, 
                 action_dim,
                 action_num,
                 starting_expl_threshold = 1.0,
                 expl_decay_factor = 0.999, 
                 min_expl_threshold = 0.01
                 ):
         
        self.action_dim = action_dim
        self.action_num= action_num
        
        self.current_exploration = starting_expl_threshold
        self.expl_decay_factor = expl_decay_factor
        self.min_expl_threshold = min_expl_threshold        
        
    def get_action(self, q_network, state):
        
        # TODO: test the functionality 
        
        # get a random number
        random_num = np.random.uniform(0,1,1)[0]
        
        # get a random action
        if random_num <= self.current_exploration:
            action = np.random.randint(0, self.action_num, size=self.action_dim)
            
        # take the action(s) with the max Q value(s)
        else:
            with torch.no_grad():
                action = q_network(state).cpu().data.numpy()
                
        return action
                
                
if __name__ == "__main__":    
    
    exp = epsilon_greedy(2, np.array([2, 3]))
    
    exp.get_action()
    
    
    
    