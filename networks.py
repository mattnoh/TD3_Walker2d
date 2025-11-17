"""
Actor and Critic neural networks for TD3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Actor network with LayerNorm for stability"""
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x


class Critic(nn.Module):
    """Twin critic networks with LayerNorm"""
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, 400)
        self.ln4 = nn.LayerNorm(400)
        self.fc5 = nn.Linear(400, 300)
        self.ln5 = nn.LayerNorm(300)
        self.fc6 = nn.Linear(300, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.xavier_uniform_(self.fc5.weight)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        self.fc6.weight.data.uniform_(-3e-3, 3e-3)
        self.fc6.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.ln1(self.fc1(sa)))
        q1 = F.relu(self.ln2(self.fc2(q1)))
        q1 = self.fc3(q1)
        
        q2 = F.relu(self.ln4(self.fc4(sa)))
        q2 = F.relu(self.ln5(self.fc5(q2)))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.ln1(self.fc1(sa)))
        q1 = F.relu(self.ln2(self.fc2(q1)))
        q1 = self.fc3(q1)
        return q1