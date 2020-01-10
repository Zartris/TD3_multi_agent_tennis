import torch.nn as nn
from skimage.io import imsave

from ..utils import *


class AgentBase:
    def __init__(self,
                 agent_name: str,
                 save_path: Path,
                 state_normalizer,
                 log_level: int,
                 seed: int
                 ):
        # Load, save, logging options
        self.name = agent_name

        self.save_path = save_path
        if self.save_path is None:
            self.save_path = Path(os.getcwd(), "..", "save_folder", get_time_str())

        self.logger = get_logger(tag=self.name, log_level=log_level)

        # Agent options
        self.state_normalizer = state_normalizer
        self.total_steps = 0  # Will be incremented in step function

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

    def save(self, model_name: str, model: nn.Module):
        torch.save(model.state_dict(), str(Path(self.save_path, '%s.model' % model_name)))

    def save_stats(self, filename: str):
        with open(str(Path(self.save_path, '%s.stats' % filename)), 'wb') as f:
            pickle.dump(self.state_normalizer.state_dict(), f)

    def load_state_dict(self, model_name: str, load_path: Path = None) -> dict:
        if load_path is None:
            load_path = self.save_path
        state_dict = torch.load(str(Path(load_path, '%s.model' % model_name)))
        return state_dict

    def load_stats(self, filename: str, load_path: Path = None):
        if load_path is None:
            load_path = self.save_path
        with open(str(Path(load_path, '%s.stats' % filename)), 'rb') as f:
            self.state_normalizer.load_state_dict(pickle.load(f))

    def save_all(self):
        """
        This is to be overwritten
        """
        raise NotImplementedError

    def load_all(self):
        """
        This is to be overwritten
        """
        raise NotImplementedError

    def eval_step(self, state):
        """
        This is to be overwritten
        """
        raise NotImplementedError

    def eval_episode(self, env):
        """
        TODO: add return value
        :param env: The game env we are evaluating on
        :return:
        """
        state = env.reset()
        while True:
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']  # TODO: Might be to hardcoded here
            if ret is not None:
                break
        return ret

    def eval_episodes(self, env, total_episodes):
        """
        TODO: add return value
        :param env: The evaluation environment
        :param total_episodes: amount of episodes to evaluate over
        :return:
        """
        episodic_returns = []
        for ep in range(total_episodes):
            total_rewards = self.eval_episode(env=env)
            episodic_returns.append(np.sum(total_rewards))
        # log eval:
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns),
            np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    @staticmethod
    def update_target_model(local_model: nn.Module, target_model: nn.Module):
        """ Hard update model parameters.
            Copying the current weights from local_model to the target_model.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    @staticmethod
    def soft_update(local_model: nn.Module, target_model: nn.Module, tau: float):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def record_episode(self, env, save_dir: Path = None):
        if save_dir is None:
            count = 0
            save_dir = Path(self.save_path, "record_episodes", "rec" + str(count))
            while save_dir.exists():
                count += 1
                save_dir = Path(self.save_path, "record_episodes", "rec" + str(count))
            save_dir.mkdir(parents=True)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, save_dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        """
        This is to be overwritten
        :param state:
        :return:
        """
        raise NotImplementedError

    # For DMControl
    @staticmethod
    def record_obs(env, save_dir: Path, steps: int):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (str(save_dir), steps), obs)
