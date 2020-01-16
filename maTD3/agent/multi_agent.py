from maTD3.agent.AgentBase import AgentBase
from maTD3.utils import *


class MultiAgent(AgentBase):
    def __init__(self,
                 agent_name: str,
                 agents: list,
                 save_path: Path = None,
                 state_normalizer=RescaleNormalizer(),  # Todo: implement this
                 log_level: int = 0,  # 0 Equal to log everything
                 seed: int = 0):
        super().__init__(agent_name, save_path, state_normalizer, log_level, seed)
        self.agents = agents

    def act(self, states, add_noise=True):
        """Returns actions for given state as per agent."""
        actions = np.asarray([agent.act(state, add_noise=add_noise) for agent, state in zip(self.agents, states)])
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and uses samples from buffer to learn."""
        self.total_steps += 1
        for idx, (agent, state, action, reward, next_state, done) in enumerate(
                zip(self.agents, states, actions, rewards, next_states, dones)):
            # self.add_to_memory(state, action, reward, next_state, done, agent_idx=idx)
            agent.step(state, action, reward, next_state, done)
        # for agent in self.agents:
        #     agent.step(self.shared_memory)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def add_to_memory(self, states, actions, rewards, next_states, dones, agent_idx=None, error=None):
        """
        allows us to add experiences from outside this class, without stepping
        """
        for idx, (agent, state, action, reward, next_state, done) in enumerate(
                zip(self.agents, states, actions, rewards, next_states, dones)):
            agent.add_to_memory(state, action, reward, next_state, done, agent_idx=idx)

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
