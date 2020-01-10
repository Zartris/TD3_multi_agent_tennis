from collections import namedtuple

import numpy as np
import torch

from TD3.replay_buffers.RB_base import RBBase
from TD3.replay_buffers.sumtree import SumTree


class PrioritizedReplayBuffer(RBBase):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, capacity, batch_size, seed, action_size, epsilon=.001, alpha=.6, beta=.4,
                 beta_increase=1e-3, absolute_error_upper=3):
        """
        :param capacity: Max amount of experience saved in the structure
        :param epsilon: small value to insure all probabilities is not 0
        :param alpha: introduces some randomness and to insure we don't train the same experience and overfit
                      alpha=1 means greedy selecting the experience with highest priority
                      alpha=0 means pure uniform randomness
        :param beta: controls how much IS w affect learning
                     beta>=0, starts close to 0 and get closer and closer to 1
                     because these weights are more important in the end of learning when our q-values
                     begins to convert
        :param beta_increase: is the increase in beta for each sampling. 0.001 = 1e-3
        """
        super().__init__(action_size, capacity, batch_size, seed)
        # init
        self.memory_tree = SumTree(self.capacity, seed)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.absolute_error_upper = absolute_error_upper

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory_tree)

    def is_full(self):
        return len(self.memory_tree) >= self.memory_tree.capacity

    def is_full_enough(self):
        return len(self.memory_tree) >= self.batch_size

    def sample(self):
        """
                - First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
                - Then a value is uniformly sampled from each range.
                - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
                - Then, we calculate IS weights for each minibatch element.

                The difference here from the last structure is that we need to move to device
                in the method calling this function.

                so an example:
                idxs, experiences, is_weights = self.memory.sample(BATCH_SIZE)

                states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(device)
                actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).float().to(device)
                rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(device)
                next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(device)
                dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

                is_weights =  torch.from_numpy(is_weights).float().to(device)
                """

        minibatch = []

        idxs = np.empty((self.batch_size,), dtype=np.int32)
        is_weights = np.empty((self.batch_size,), dtype=np.float32)

        # Calculate the priority segment
        # Divide the Range[0, ptotal] into n ranges
        priority_segment = self.memory_tree.total_priority / self.batch_size  # priority segment

        # Increase the beta each time we sample a new minibatch
        self.beta = np.amin([1., self.beta + self.beta_increase])  # max = 1

        for i in range(self.batch_size):
            """
            A value is uniformly sampled from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)

            # This while is to counter that we find a leaf that is not populated yet.
            # It happens when the buffer size is very large.
            data, index, priority = 0, 0, 0
            while data == 0:
                value = np.random.uniform(a, b)

                """
                Experience that corresponds to each value is retrieved
                """
                index, priority, data = self.memory_tree.get_leaf(value)

            # P(i) = p_i**a / sum_k p_k**a
            sampling_probabilities = priority / self.memory_tree.total_priority

            # (1/N * 1/P(i))**b and to normalize it we divide with max_weight
            # So is_weights[i] = (1/N * 1/P(i))**b
            is_weights[i] = np.power(self.batch_size * sampling_probabilities, -self.beta)

            idxs[i] = index
            minibatch.append(data)
        is_weights /= is_weights.max()
        return idxs, self.transform_batch(minibatch), is_weights

    def transform_batch(self, experiences):
        states = torch.from_numpy(np.vstack([e[0] for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences if e is not None]).astype(np.uint8)).float().to(
            self.device)
        return states, actions, rewards, next_states, dones

    def add(self, state, action, reward, next_state, done, agent_idx=None, error=None):
        experience = self.experience(state, action, reward, next_state, done)
        if error is None:
            priority = np.amax(self.memory_tree.tree[-self.memory_tree.capacity:])
            if priority == 0:
                priority = self.absolute_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        self.memory_tree.add(experience, priority)

    def update_memory_tree(self, idxs, errors):
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)

        for idx, p in zip(idxs, ps):
            self.memory_tree.update(idx, p)
