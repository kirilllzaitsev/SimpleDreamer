
import gym
import numpy as np
import torch
from gym import spaces
from legged_gym.envs import task_registry


def get_env(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    env = IsaacGymWrapper(env)
    return env, env_cfg


class IsaacGymWrapper(gym.Env):
    # maps isaacgym interface to gym interface
    def __init__(
        self,
        isaac_env,
        action_space_low=-2.0,
        action_space_high=2.0,
        observation_space_low=-10.0,
        observation_space_high=10.0,
    ):
        self.isaac_env = isaac_env
        self.action_space = spaces.Box(
            low=np.array(action_space_low).repeat(self.isaac_env.num_actions),
            high=np.array(action_space_high).repeat(self.isaac_env.num_actions),
            dtype=np.float32,
            shape=(self.isaac_env.num_actions,),
        )
        self.observation_space = spaces.Box(
            low=np.array(observation_space_low).repeat(self.isaac_env.num_obs),
            high=np.array(observation_space_high).repeat(self.isaac_env.num_obs),
            dtype=np.float32,
            shape=(self.isaac_env.num_obs,),
        )

    def reset(self):
        state, _ = self.isaac_env.reset()
        return np.array(state.cpu(), dtype=np.float32)

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        state, privileged_state, reward, done, info = self.isaac_env.step(action)
        return (
            np.array(state.cpu(), dtype=np.float32),
            (reward.cpu().squeeze()),
            (done.cpu().squeeze()),
            info,
        )

    def render(self, mode="human"):
        self.isaac_env.render(mode=mode)

    def close(self):
        self.isaac_env.close()

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            self.isaac_env.seed(seed)
