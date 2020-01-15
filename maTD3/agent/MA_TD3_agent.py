import torch.nn.functional as F
import torch.optim as optim

from maTD3.agent.AgentBase import AgentBase
from maTD3.model.twin_ac_model import TwinCritic, TD3Actor
from maTD3.replay_buffers.replay_buffer import ReplayBuffer
from maTD3.utils import *

N_STEP = 3


### NOTES:
# PER           https://arxiv.org/pdf/1707.08817.pdf
# TD3 medium:   https://towardsdatascience.com/td3-learning-to-run-with-ai-40dfc512f93
# TD3 Extra:    https://spinningup.openai.com/en/latest/algorithms/td3.html#background
class MATD3Agent(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self,
                 agent_name: str,
                 actor_func,
                 twin_critic,
                 twin_critic_target,
                 critic_optimizer,
                 replay_buffer,
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
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 exploration_noise: float = 0.3,
                 noise_reduction_factor: float = 0.99,
                 noise_scalar_init: float = 2.0):
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
        self.actor = actor_func().to(self.device)
        self.actor_target = actor_func().to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Using Twin Critic Network (w/ Target Network), we are combining the two critics into the same network
        self.twin_critic = twin_critic
        self.twin_critic_target = twin_critic_target
        self.critic_optimizer = critic_optimizer

        # Learning count:
        self.train_count = 0

        # Replay buffer:
        self.memory = replay_buffer

        # Amount of training rounds:
        self.train_delay = train_delay
        self.steps_before_train = steps_before_train
        self.train_iterations = train_iterations
        self.discount = discount
        self.tau = tau
        self.lr_actor = lr_actor
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.exploration_noise = exploration_noise
        self.noise_reduction_factor = noise_reduction_factor
        self.noise_scalar_init = noise_scalar_init
        self.noise_scalar = noise_scalar_init

    def reset(self):
        self.noise_scalar = self.noise_scalar_init

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).cpu().data.numpy()
        self.actor.train()
        if add_noise:
            actions += np.random.normal(0, self.action_val_high * self.exploration_noise,
                                        self.action_size) * self.noise_scalar
            self.noise_scalar *= self.noise_reduction_factor
        return np.clip(actions, self.action_val_low, self.action_val_high)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.total_steps += 1
        self.memory.add(state, action, reward, next_state, done)
        # Time to train.
        # We want to train the critics every step but delay the actor update for self.train_delay of time.
        # Learn, if enough samples are available in memory
        if self.memory.is_full_enough():
            if self.total_steps % self.steps_before_train == 0:
                for _ in range(self.train_iterations):
                    idxs, experiences, is_weights = self.memory.sample()
                    self.learn(experiences)

    def step_shared(self, shared_memory):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.total_steps += 1
        # Time to train.
        # We want to train the critics every step but delay the actor update for self.train_delay of time.
        # Learn, if enough samples are available in memory
        if shared_memory.is_full_enough():
            if self.total_steps % self.steps_before_train == 0:
                for _ in range(self.train_iterations):
                    idxs, experiences, is_weights = shared_memory.sample()
                    self.learn(experiences)

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
        # with torch.no_grad():
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
    agent = MATD3Agent(agent_name="TD3_agent",
                       actor=actor,
                       twin_critic=twin_critic,
                       action_size=action_size,
                       action_val_high=action_val_max,
                       action_val_low=action_val_min)
