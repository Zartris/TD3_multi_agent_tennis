from collections import namedtuple, deque

import torch

from TD3.replay_buffers.prioritized_experience_replay import PrioritizedReplayBuffer


class PerNStep(PrioritizedReplayBuffer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self, capacity, batch_size, state_size, action_size, seed=None, epsilon=.0001, alpha=.6, beta=.4,
                 beta_increase=1e-3, absolute_error_upper=3, n_step=3, gamma=.99):
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
        super().__init__(capacity, batch_size, seed, action_size, epsilon, alpha, beta, beta_increase,
                         absolute_error_upper)
        # init
        self.experience = namedtuple("Experience",
                                     field_names=["timestep", "state", "action", "reward", "next_state", "done"])

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

    def add(self, state, action, reward, next_state, done, agent_idx=None, error=None):
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
