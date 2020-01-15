import os
from collections import deque
from pathlib import Path

import torch
import numpy as np
from unityagents import UnityEnvironment

from testFolder.RB import ReplayBuffer
from testFolder.agent import TD3Agent

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
LEARN_EVERY = 4        # how often to learn from the experience
UPDATE_EVERY = 2        # how often to update the target network
random_seed = 0



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gamefile = Path(os.getcwd(),"..", 'Tennis', 'Tennis.exe')
env = UnityEnvironment(file_name=str(gamefile), seed=random_seed)
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

public_replay_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

agents = []

for i in range(num_agents):
    agents.append(TD3Agent(state_size=state_size,
                           action_size=action_size,
                           max_action=1.0,
                           min_action=-1.0,
                           memory=public_replay_buffer,
                           random_seed=0))


def td3(n_episodes=2000, max_t=100000):
    scores_deque = deque(maxlen=100)
    solved = False
    total_scores = []
    highest_score = float('-inf')
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        # Initialize noise reduction
        noise_degree = 2.0
        noise_decay = 0.999
        while True:
            actions = []
            for i in range(len(states)):
                actions.append(agents[i].act(states[i], True, noise_degree))
            noise_degree *= noise_decay
            env_info = env.step(actions)[brain_name]  # send all actions to the environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done

            for i in range(num_agents):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i])

            states = next_states
            scores += rewards
            if np.any(dones):
                break
        scores_deque.append(np.max(scores))
        total_scores.append(scores)
        mean_score = np.mean(scores_deque)
        print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, mean_score, np.max(scores)),
              end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score >= 0.5 and mean_score > highest_score and i_episode % 10 == 0:
            torch.save(agents[0].actor_local.state_dict(), 'actor1.pth')
            torch.save(agents[1].actor_local.state_dict(), 'actor2.pth')
            torch.save(agents[0].critic_local1.state_dict(), 'critic1.pth')
            torch.save(agents[0].critic_local2.state_dict(), 'critic2.pth')
            print('\rSave at {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score >= 0.5 and solved == False:
            solved = True
            torch.save(agents[0].actor_local.state_dict(), 'actor1.pth')
            torch.save(agents[1].actor_local.state_dict(), 'actor2.pth')
            torch.save(agents[0].critic_local1.state_dict(), 'critic1.pth')
            torch.save(agents[0].critic_local2.state_dict(), 'critic2.pth')
            print('\rSolved at Episode {} !\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        highest_score = max(highest_score, mean_score)
    return total_scores


scores = td3()
