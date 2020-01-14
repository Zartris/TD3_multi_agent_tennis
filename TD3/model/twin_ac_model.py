import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # state = state.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class TD3Actor(Actor):
    """ Building a TD3 actor model upon the base actor.
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            max_action (float): highest action to take <-- New
            seed (int): Random seed
            h1_units (int): Number of nodes in first hidden layer
            h2_units (int): Number of nodes in second hidden layer

        Return:
            action output of network with tanh activation
    """

    def __init__(self, state_size, action_size, seed, max_action, fc1_units=512, fc2_units=256):
        super().__init__(state_size, action_size, seed, fc1_units=fc1_units, fc2_units=fc2_units)
        self.max_action = max_action

    def forward(self, state):
        x = super().forward(state)
        # TODO: Check why we multiply with max action?
        return self.max_action * x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.bn1(self.fcs1(state)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TwinCritic(nn.Module):
    """TwinCritic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(TwinCritic, self).__init__()
        self.seed = torch.manual_seed(seed)

        # First Critic C1
        # All the layers (we need them as variables to reset each layer)
        self.c1_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.c1_bn1 = nn.BatchNorm1d(fc1_units)
        self.c1_fc2 = nn.Linear(fc1_units, fc2_units)
        self.c1_fc3 = nn.Linear(fc2_units, 1)
        # Add them into a Sequential function
        self.critic1 = nn.Sequential(
            self.c1_fc1,
            self.c1_bn1,
            nn.ReLU(),
            self.c1_fc2,
            nn.ReLU(),
            self.c1_fc3
        )

        # Second Critic C2 ( Have to be the same architecture)
        self.c2_fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.c2_bn1 = nn.BatchNorm1d(fc1_units)
        self.c2_fc2 = nn.Linear(fc1_units, fc2_units)
        self.c2_fc3 = nn.Linear(fc2_units, 1)
        self.critic2 = nn.Sequential(
            self.c2_fc1,
            self.c2_bn1,
            nn.ReLU(),
            self.c2_fc2,
            nn.ReLU(),
            self.c2_fc3
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Reset C1
        self.c1_fc1.weight.data.uniform_(*hidden_init(self.c1_fc1))
        self.c1_fc2.weight.data.uniform_(*hidden_init(self.c1_fc2))
        self.c1_fc3.weight.data.uniform_(-3e-3, 3e-3)
        # Reset C2
        self.c2_fc1.weight.data.uniform_(*hidden_init(self.c2_fc1))
        self.c2_fc2.weight.data.uniform_(*hidden_init(self.c2_fc2))
        self.c2_fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        sa = torch.cat((state, action), dim=1)
        q1 = self.critic1(sa)
        q2 = self.critic2(sa)
        return q1, q2

    def Q1(self, state, action):
        """ To get only Q1 value"""
        sa = torch.cat((state, action), dim=1)
        q1 = self.critic1(sa)
        return q1
