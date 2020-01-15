import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
import matplotlib.pyplot as plt

from testFolder.model import Critic, Actor
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
LEARN_EVERY = 4        # how often to learn from the experience
UPDATE_EVERY = 2        # how often to update the target network
random_seed = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent():
    """Interacts with and learns from the environment."""
    # Shared crtic among all agents
    critic_local1 = None
    critic_target1 = None
    critic_optimizer1 = None

    critic_local2 = None
    critic_target2 = None
    critic_optimizer2 = None

    def __init__(self, state_size, action_size, max_action, min_action, memory, random_seed, noise=0.2, noise_std=0.3,
                 noise_clip=0.5):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            max_action (ndarray): the maximum valid value for each action vector
            min_action (ndarray): the minimum valid value for each action vector
            random_seed (int): random seed
            noise (float): the range to generate random noise while learning
            noise_std (float): the range to generate random noise while performing action
            noise_clip (float): to clip random noise into this range
        """
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.min_action = min_action
        self.noise = noise
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        if TD3Agent.critic_local1 is None:
            TD3Agent.critic_local1 = Critic(state_size, action_size).to(device)
        if TD3Agent.critic_target1 is None:
            TD3Agent.critic_target1 = Critic(state_size, action_size).to(device)
            TD3Agent.critic_target1.load_state_dict(self.critic_local1.state_dict())
        if TD3Agent.critic_optimizer1 is None:
            TD3Agent.critic_optimizer1 = optim.Adam(self.critic_local1.parameters(), lr=LR_CRITIC)
        self.critic_local1 = TD3Agent.critic_local1
        self.critic_target1 = TD3Agent.critic_target1
        self.critic_optimizer1 = TD3Agent.critic_optimizer1

        if TD3Agent.critic_local2 is None:
            TD3Agent.critic_local2 = Critic(state_size, action_size).to(device)
        if TD3Agent.critic_target2 is None:
            TD3Agent.critic_target2 = Critic(state_size, action_size).to(device)
            TD3Agent.critic_target2.load_state_dict(self.critic_local2.state_dict())
        if TD3Agent.critic_optimizer2 is None:
            TD3Agent.critic_optimizer2 = optim.Adam(self.critic_local2.parameters(), lr=LR_CRITIC)
        self.critic_local2 = TD3Agent.critic_local2
        self.critic_target2 = TD3Agent.critic_target2
        self.critic_optimizer2 = TD3Agent.critic_optimizer2

        # Shared Replay memory
        self.memory = memory

        self.lt_step = 0
        self.ut_step = 0

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory"""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn every LEARN_EVERY times
        self.lt_step = (self.lt_step + 1) % LEARN_EVERY
        if self.lt_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                # update network UPDATE_EVERY times
                for t in range(UPDATE_EVERY):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA)

    def act(self, state, noise_reduction=0.0, add_noise=True):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (ndarray): the observation of current state
            noise_reduction (float): the number for adjusting noise while training
            add_noise (bool): whether to add noise in action
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if add_noise:
            # Generate a random noise
            noise = np.random.normal(0, self.noise_std, size=self.action_size)
            noise *= noise_reduction
            # Add noise to the action for exploration
            action = (action + noise).clip(self.min_action, self.max_action)
        self.actor_local.train()
        return action

    def learn(self, experiences, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """

        state, action, reward, next_state, done = experiences

        action_ = action.cpu().numpy()

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_state)

        # Generate a random noise
        noise = torch.FloatTensor(action_).data.normal_(0, self.noise).to(device)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        actions_next = (actions_next + noise).clamp(self.min_action, self.max_action)

        Q1_targets_next = self.critic_target1(next_state, actions_next)
        Q2_targets_next = self.critic_target2(next_state, actions_next)

        Q_targets_next = torch.min(Q1_targets_next, Q2_targets_next)
        # Compute Q targets for current states (y_i)
        Q_targets = reward + (gamma * Q_targets_next * (1 - done)).detach()
        # Compute critic loss
        Q1_expected = self.critic_local1(state, action)
        Q2_expected = self.critic_local2(state, action)
        critic_loss1 = F.mse_loss(Q1_expected, Q_targets)
        critic_loss2 = F.mse_loss(Q2_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer1.zero_grad()
        critic_loss1.backward()
        self.critic_optimizer1.step()

        self.critic_optimizer2.zero_grad()
        critic_loss2.backward()
        self.critic_optimizer2.step()

        self.ut_step = (self.ut_step + 1) % UPDATE_EVERY
        if self.ut_step == 0:
            # ---------------------------- update actor ---------------------------- #
            # Compute actor loss
            actions_pred = self.actor_local(state)
            actor_loss = -self.critic_local1(state, actions_pred).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local1, self.critic_target1, TAU)
            self.soft_update(self.critic_local2, self.critic_target2, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)