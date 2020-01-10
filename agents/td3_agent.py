import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.model import Actor, TwinCritic
from replay_buffers.per_nstep import PerNStep
from replay_buffers.replay_buffer import ReplayBuffer

N_STEP = 3


### NOTES:
# PER           https://arxiv.org/pdf/1707.08817.pdf
# TD3 medium:   https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
# TD3 Extra:    https://spinningup.openai.com/en/latest/algorithms/td3.html#background
class TD3Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,  # These are just default parameters, check results\solved\model_test for used hyper-parameters
                 state_size: int,  #
                 action_size: int,  #
                 action_val_high: float,  #
                 action_val_low: float,  #
                 random_seed: int = 0,  #
                 train_delay: int = 2,  #
                 steps_before_train=1,  # Number of steps between training seasons, set to 1 if train after each step
                 train_iterations=1,    # Number of batches train each training season
                 buffer_size: int = 2 ** 20,
                 batch_size: int = 512,
                 discount: float = 0.99,  # Discount factor
                 tau: float = 1e-3,
                 lr_actor: float = 4e-4,
                 lr_critic: float = 4e-4,
                 weight_decay: float = 1e-6,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 exploration_noise: float = 0.1,
                 per: bool = False,  # Not implemented yet, but will come.
                 model_dir: str = os.getcwd()  # Save\load model dir
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_val_high = action_val_high
        self.action_val_low = action_val_low
        self.seed = random.seed(random_seed)
        self.model_dir = model_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Should have been parameters, but now they are here.
        fc1_units = 400
        fc2_units = 300

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed=random_seed, fc1_units=fc1_units,
                                 fc2_units=fc2_units).to(
            self.device)
        self.actor_target = Actor(state_size, action_size, seed=random_seed, fc1_units=fc1_units,
                                  fc2_units=fc2_units).to(
            self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        # Using Twin Critic Network (w/ Target Network), we are combining the two critics into the same network
        self.twin_critic_local = TwinCritic(state_size, action_size, seed=random_seed, fc1_units=fc1_units,
                                            fc2_units=fc2_units).to(
            self.device)
        self.twin_critic_target = TwinCritic(state_size, action_size, seed=random_seed, fc1_units=fc1_units,
                                             fc2_units=fc2_units).to(self.device)
        self.critic_optimizer = optim.Adam(self.twin_critic_local.parameters(), lr=lr_critic,
                                           weight_decay=weight_decay)  # TODO: Test if wdcay helps learning else lower lr

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Combined replay buffer:
        self.per = per
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        if per:
            self.memory = PerNStep(buffer_size, batch_size, state_size, seed=random_seed, n_step=N_STEP)
        else:
            self.memory = ReplayBuffer(action_size, buffer_size=buffer_size, batch_size=batch_size, seed=random_seed)
        # Learning count:
        self.step_count = 0
        self.train_count = 0

        # Amount of training rounds:
        self.train_delay = train_delay
        self.steps_before_train = steps_before_train
        self.train_iterations = train_iterations
        self.discount = discount
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.step_count += 1
        # Save experience / reward
        agent_idx = 0  # This is needed if we are ever to use N-step but currently just here for nothing
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(agent_idx, state, action, reward, next_state, done)
            agent_idx += 1

        # Time to train.
        # We want to train the critics every step but delay the actor update for self.train_delay of time.
        # Learn, if enough samples are available in memory

        if self.step_count % self.steps_before_train == 0:
            self.step_count = 0
            if len(self.memory) > self.batch_size:
                for _ in range(self.train_iterations):
                    self.learn()

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            # OBSERVATION: Use gaussian noise instead of OUNoise, this improve performance a lot!
            actions += np.random.normal(0, self.action_val_high * self.exploration_noise, self.action_size)
        return np.clip(actions, self.action_val_low, self.action_val_high)

    def reset(self):
        self.noise.reset()

    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
                Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
                where:
                    actor_target(state) -> action
                    critic_target(state, action) -> Q-value
                Params
                ======
                    experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
                    gamma (float): discount factor
                """
        self.train_count += 1
        # Get a batch of experiences:
        idxs, experiences, is_weights = self.memory.sample()  # take a batch to train the critics with
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        # Compute the Target Q (min of Q1 and Q2)
        # TODO: Question, is this nessary? we are never using the gradient anyway. Sure for performance.
        with torch.no_grad():
            # performing policy smooth, by adding noise, to reduce variance.
            noise = (torch.randn_like(actions) * self.policy_noise)
            # clamping the noise to keep the target value close to original action
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            # using actor target to get next state and add noise
            next_actions = (self.actor_target(next_states) + noise)
            # claping to make sure they are within action value range
            next_actions = next_actions.clamp(self.action_val_low, self.action_val_high)

            # Compute the target Q value:
            Q1_targets, Q2_targets = self.twin_critic_target(next_states, next_actions)
            target_Q = torch.min(Q1_targets, Q2_targets)
            target_Q = rewards + (self.discount * target_Q * (1 - dones))

        # Compute critic loss
        expected_Q1, expected_Q2 = self.twin_critic_local(states, actions)  # let it compute gradient
        critic_loss = F.mse_loss(expected_Q1, target_Q) + F.mse_loss(expected_Q2, target_Q)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.twin_critic_local.parameters(), 1)
        self.critic_optimizer.step()
        # Delayed training of actor network:
        if self.train_count % self.train_delay == 0:
            # ---------------------------- update actor ---------------------------- #
            self.train_count = 0
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actor_loss = -self.twin_critic_local.Q1(states, actions_pred).mean()  # Minus is for maximizing
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.twin_critic_local, self.twin_critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)

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

    def save(self):
        torch.save(self.actor_local.state_dict(), str(Path(self.model_dir, 'actor.pth')))
        torch.save(self.actor_target.state_dict(), str(Path(self.model_dir, 'actor_target.pth')))
        torch.save(self.twin_critic_local.state_dict(), str(Path(self.model_dir, 'twin_critic.pth')))
        torch.save(self.twin_critic_target.state_dict(), str(Path(self.model_dir, 'twin_critic_target.pth')))

    def load(self):
        self.actor_local.load_state_dict(torch.load(str(Path(self.model_dir, 'actor.pth'))))
        self.actor_target.load_state_dict(torch.load(str(Path(self.model_dir, 'actor_target.pth'))))
        self.twin_critic_local.load_state_dict(torch.load(str(Path(self.model_dir, 'twin_critic.pth'))))
        self.twin_critic_target.load_state_dict(torch.load(str(Path(self.model_dir, 'twin_critic_target.pth'))))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
