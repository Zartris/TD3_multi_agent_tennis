from collections import namedtuple, deque

import numpy as np
import torch


class PerNStep:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, capacity, batch_size, state_size, seed=None, epsilon=.0001, alpha=.6, beta=.4,
                 beta_increase=1e-3,
                 absolute_error_upper=3, n_step=3, gamma=.99):
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
        :param absolute_error_upper: Setting a cap on how big an error (priority) can be.
        :param n_step: store the most recent n-step transitions or experiences instead of the default 1.
        :param gamma: This is the discount value
        """
        ## Just like PER
        # seeding
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        # init
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory_tree = SumTree(self.capacity, seed)
        self.experience = namedtuple("Experience",
                                     field_names=["timestep", "state", "action", "reward", "next_state", "done"])

        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.beta_increase = beta_increase
        self.absolute_error_upper = absolute_error_upper
        self.seed = seed

        ## N-Step
        self.t = 0  # Internal time step counter
        self.n_step = n_step
        self.n_step_buff = {}
        self.gamma = gamma
        self.blank_experience = self.experience(timestep=0,
                                                state=torch.zeros(state_size, dtype=torch.float64),
                                                action=None,
                                                reward=0,
                                                next_state=torch.zeros(state_size, dtype=torch.float64),
                                                done=False)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory_tree)

    def is_full(self):
        return len(self.memory_tree) >= self.memory_tree.capacity

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
            leaf_index, data_index, priority, data = 0, 0, 0, 0
            while data == 0:
                value = np.random.uniform(a, b)

                """
                Experience that corresponds to each value is retrieved
                """
                leaf_index, data_index, priority, data = self.memory_tree.get_leaf(value)

            # P(i) = p_i**a / sum_k p_k**a
            sampling_probabilities = priority / self.memory_tree.total_priority

            # (1/N * 1/P(i))**b and to normalize it we divide with max_weight
            # So is_weights[i] = (1/N * 1/P(i))**b
            is_weights[i] = np.power(self.batch_size * sampling_probabilities, -self.beta)

            idxs[i] = leaf_index
            minibatch.append(data)
        is_weights /= is_weights.max()
        return idxs, minibatch, is_weights

    def add(self, agent_idx, state, action, reward, next_state, done, error=None):
        """Customized to more than one agent"""
        exp = self.experience(self.t, torch.from_numpy(state), action, reward, torch.from_numpy(next_state), done)
        if agent_idx not in self.n_step_buff:
            self.n_step_buff[agent_idx] = deque(maxlen=self.n_step)
        self.n_step_buff[agent_idx].append(exp)
        if agent_idx == 0:
            self.t = (0 if done else self.t + 1)
        if len(self.n_step_buff[agent_idx]) < self.n_step:
            return None
        exp, priority = self._get_n_step_info(self.n_step_buff[agent_idx], self.gamma)
        priority = min((abs(priority) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        self.memory_tree.add(exp, priority)

    def update_memory_tree(self, idxs, errors):
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)

        for idx, p in zip(idxs, ps):
            self.memory_tree.update(idx, p)

    def _get_n_step_info(self, n_step_buff, gamma):
        timestep, org_state, org_action, _, _, _ = n_step_buff[0]
        relevant_transitions = []
        for transition in list(n_step_buff):
            if timestep == transition.timestep:
                relevant_transitions.append(transition)
                timestep += 1
            else:
                break
        # Take last element in deque and add the reward
        rew, n_state, done = relevant_transitions[-1][-3:]
        for transition in reversed(relevant_transitions[:-1]):
            reward, n_s, d = transition.reward, transition.next_state, transition.done
            rew = reward + gamma * rew * (1 - done)
            n_state, done = (n_s, d) if d else (n_state, done)
        return self.experience(timestep, org_state, org_action, rew, n_state, done), rew


# https://github.com/Kaixhin/Rainbow/blob/master/memory.py
class SumTree:
    def __init__(self, capacity, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.capacity = int(capacity)
        assert self.is_power_of_2(self.capacity), "Capacity must be power of 2." + str(capacity)
        # pointer to current index in data map.
        self.data_pointer = 0
        self.data = np.zeros(capacity, dtype=object)
        self.data_length = 0
        # Priority tree.
        self.tree = np.zeros(2 * capacity - 1)

    def __len__(self):
        return self.data_length

    def add(self, data, priority):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        # Update data frame
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(tree_index, priority)
        # Add 1 to data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.data_length < self.capacity:
            self.data_length += 1

    def update(self, tree_index, priority):
        # TODO: This is the time killer!
        # change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def get_leaf(self, value):
        parent_index = 0  # root
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, data_index, self.tree[leaf_index], self.data[data_index]

    # Returns data given a data index
    def get_data(self, data_index):
        return self.data[data_index % self.capacity]

    @staticmethod
    def is_power_of_2(n):
        return ((n & (n - 1)) == 0) and n != 0

    @property
    def total_priority(self):
        return self.tree[0]  # the root
