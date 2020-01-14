from TD3.agent.AgentBase import AgentBase
from TD3.utils import *


class MultiAgent(AgentBase):
    def __init__(self,
                 agent_name: str,
                 agents: list,
                 shared_replay_buffer,
                 save_path: Path = None,
                 state_normalizer=RescaleNormalizer(),  # Todo: implement this
                 log_level: int = 0,  # 0 Equal to log everything
                 seed: int = 0):
        super().__init__(agent_name, save_path, state_normalizer, log_level, seed)

        self.agents = agents
        self.shared_memory = shared_replay_buffer

    def act(self, states, add_noise=True):
        """Returns actions for given state as per agent."""
        actions = np.asarray([agent.act(state, add_noise=add_noise) for agent, state in zip(self.agents, states)])
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and uses samples from buffer to learn."""
        self.total_steps += 1
        for idx, (state, action, reward, next_state, done) in enumerate(
                zip(states, actions, rewards, next_states, dones)):
            self.add_to_memory(state, action, reward, next_state, done, agent_idx=idx)

        if self.shared_memory.is_full_enough():
            
        pass

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

    def learn(self):
        """
        !Only supports normal Replay buffer!
        :return:
        """
        for agent in self.agents:
            idxs, experiences, is_weights = self.shared_memory.sample()
            agent.learn(experiences)

    def reset(self):
        for agent in self.agent:
            agent.reset()

    def add_to_memory(self, state, action, reward, next_state, done, agent_idx=None, error=None):
        """
        allows us to add experiences from outside this class
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        self.shared_memory.add(state, action, reward, next_state, done, agent_idx=agent_idx, error=error)

    def save_all(self):
        for agent in self.agents:
            agent.save_all()

    def load_all(self):
        for agent in self.agents:
            agent.load_all()

    def eval_step(self, state):
        pass

    def record_step(self, state):
        pass
