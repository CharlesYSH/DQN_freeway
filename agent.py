#!/usr/bin/python3
#coding=utf-8

import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RandomAgent(object):
    def __init__(self):
        self.action_dim = 3
        self.interaction_steps = 0

    def select_action(self, state):
        self.interaction_steps += 1
        return torch.tensor( [random.sample([0,1,2],1)], device=device, dtype=torch.long )

    def evaluate_action(self, state):
        return torch.tensor( [random.sample([0,1,2],1)], device=device, dtype=torch.long )
