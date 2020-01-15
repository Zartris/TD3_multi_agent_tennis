import random
from collections import deque, namedtuple

import numpy as np
import torch

from maTD3.replay_buffers.RB_base import RBBase

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(RBBase):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        super().__init__(action_size=action_size, buffer_size=buffer_size, batch_size=batch_size, seed=seed)

        self.memory = deque(maxlen=self.capacity)  # internal memory (deque)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done, agent_idx=None, error=None):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        # Used to blend with PER feedback
        return None, (states, actions, rewards, next_states, dones), None

    def is_full(self):
        return len(self.memory) == self.capacity

    def is_full_enough(self):
        return len(self.memory) >= self.batch_size

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
