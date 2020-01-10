import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from TD3.agent.AgentBase import AgentBase
from TD3.model.twin_ac_model import TwinCritic, TD3Actor
from TD3.replay_buffers.replay_buffer import ReplayBuffer
from TD3.utils import *

N_STEP = 3


### NOTES:
# PER           https://arxiv.org/pdf/1707.08817.pdf
# TD3 medium:   https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
# TD3 Extra:    https://spinningup.openai.com/en/latest/algorithms/td3.html#background
class TD3Agent(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 agent_name: str,
                 actor: nn.Module,
                 twin_critic: nn.Module,
                 action_size: int,
                 action_val_high: float,
                 action_val_low: float,
                 save_path: Path = None,
                 state_normalizer=RescaleNormalizer(),  # Todo: implement this
                 log_level: int = 0,  # 0 Equal to log everything
                 seed: int = 0,
                 train_delay: int = 2,
                 steps_before_train=1,
                 train_iterations=1,
                 discount: float = 0.99,
                 tau: float = 1e-3,
                 lr_actor: float = 4e-4,
                 lr_critic: float = 4e-4,
                 weight_decay: float = 1e-6,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 exploration_noise: float = 0.1):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        super().__init__(agent_name, save_path=save_path, state_normalizer=state_normalizer, log_level=log_level,
                         seed=seed)
        self.action_size = action_size
        self.action_val_high = action_val_high
        self.action_val_low = action_val_low

        # Actor Network (w/ Target Network)
        self.actor = actor.to(self.device)
        self.actor_target = copy.deepcopy(actor).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Using Twin Critic Network (w/ Target Network), we are combining the two critics into the same network
        self.twin_critic = twin_critic.to(self.device)
        self.twin_critic_target = copy.deepcopy(twin_critic).to(self.device)
        self.critic_optimizer = optim.Adam(self.twin_critic.parameters(), lr=lr_critic,
                                           weight_decay=weight_decay)

        # Learning count:
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

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            # OBSERVATION: Use gaussian noise instead of OUNoise, this improve performance a lot!
            actions += np.random.normal(0, self.action_val_high * self.exploration_noise, self.action_size)
        return np.clip(actions, self.action_val_low, self.action_val_high)

    def learn(self, experiences):
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
        expected_Q1, expected_Q2 = self.twin_critic(states, actions)  # let it compute gradient
        critic_loss = F.mse_loss(expected_Q1, target_Q) + F.mse_loss(expected_Q2, target_Q)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.twin_critic.parameters(), 1)
        self.critic_optimizer.step()
        # Delayed training of actor network:
        if self.train_count % self.train_delay == 0:
            # ---------------------------- update actor ---------------------------- #
            self.train_count = 0
            # Compute actor loss
            actions_pred = self.actor(states)
            actor_loss = -self.twin_critic.Q1(states, actions_pred).mean()  # Minus is for maximizing
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.twin_critic, self.twin_critic_target, self.tau)
            self.soft_update(self.actor, self.actor_target, self.tau)

    def eval_step(self, state):
        pass

    def record_step(self, state):
        pass

    def save_all(self):
        super().save(self.name + "_actor", self.actor)
        super().save(self.name + "_twin_critic", self.twin_critic)
        super().save_stats(self.name + "_state_normalizer")

    def load_all(self, load_path: Path = None):
        self.actor.load_state_dict(super().load_state_dict(self.name + "_actor", load_path))
        self.actor_target.load_state_dict(super().load_state_dict(self.name + "_actor", load_path))
        self.twin_critic.load_state_dict(super().load_state_dict(self.name + "_twin_critic", load_path))
        self.twin_critic_target.load_state_dict(super().load_state_dict(self.name + "_twin_critic", load_path))
        self.load_stats(self.name + "_state_normalizer", load_path)


if __name__ == '__main__':
    # Example
    # Hardcoded for sake of the example
    action_size = 4
    state_size = 10
    buffer_size = 2 ** 10
    batch_size = 64
    seed = 0

    action_val_max = 10
    action_val_min = -10

    replay_buffer = ReplayBuffer(action_size, buffer_size, batch_size, seed)
    actor = TD3Actor(state_size, action_size, seed, max_action=action_val_max, fc1_units=400, fc2_units=300)
    twin_critic = TwinCritic(state_size, action_size, seed, fc1_units=400, fc2_units=300)
    agent = TD3Agent(agent_name="TD3_agent",
                     actor=actor,
                     twin_critic=twin_critic,
                     action_size=action_size,
                     action_val_high=action_val_max,
                     action_val_low=action_val_min)
