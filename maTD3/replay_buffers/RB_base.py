import random

import numpy as np
import torch


class RBBase:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.capacity = buffer_size
        # seeding
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_batch_size(self) -> int:
        return self.batch_size

    def is_full(self):
        raise NotImplementedError

    def is_full_enough(self):
        raise NotImplementedError

    def add(self, state, action, reward, next_state, done, agent_idx=None, error=None):
        """Add a new experience to memory."""
        raise NotImplementedError

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        raise NotImplementedError

    def update_memory_tree(self, idxs, errors):
        pass

    def __len__(self):
        """Return the current size of internal memory."""
        raise NotImplementedError
