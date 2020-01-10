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
