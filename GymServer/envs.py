
import gymnasium as gym
from gymnasium.spaces import Box
# FrameStack is now FrameStackObservation in newer gymnasium versions
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStackObservation as FrameStack
import numpy as np

# Import the environment to ensure registration
from .lunar_alighting_env import LunarAlightingEnv

# SB3 provides the modern versions of the old Baselines vectorized environments
from stable_baselines3.common.vec_env import VecEnvWrapper, SubprocVecEnv, DummyVecEnv, VecNormalize as VecNormalize_
from stable_baselines3.common.atari_wrappers import AtariWrapper

class TransposeImage(gym.ObservationWrapper):
    """
    Transposes the image from (H, W, C) to (C, H, W) for PyTorch.
    """
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low.flatten()[0],
            self.observation_space.high.flatten()[0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)

class VecFrameStack(VecEnvWrapper):
    """
    Vectorized frame stacking.
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space
        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)
        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)
        observation_space = Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, terminated, truncated, infos = self.venv.step_wait()
        # Gymnasium uses terminated/truncated instead of 'news'
        news = np.logical_or(terminated, truncated)

        self.stackedobs = np.roll(self.stackedobs, shift=-obs.shape[1], axis=1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[:, -obs.shape[1]:, ...] = obs
        return self.stackedobs, rews, terminated, truncated, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[:, -obs.shape[1]:, ...] = obs
        return self.stackedobs

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

def make_env(env_id, seed, rank):
    def _thunk():
        # Gymnasium requires a render_mode if you want to visualize later
        env = gym.make(env_id)

        # Handle Atari specific wrapping (DeepMind wrappers)
        is_atari = "AtariEnv" in str(type(env.unwrapped))
        if is_atari:
            env = AtariWrapper(env)

        # In Gymnasium, seeding is handled at reset, but we can set it here for some envs
        # Note: If this fails, move seed to env.reset(seed=seed)
        env.action_space.seed(seed + rank)

        # Handle Image Transposition for PyTorch (H,W,C -> C,H,W)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env)

        return env
    return _thunk

def make_vec_envs(env_name, seed, num_processes, num_frame_stack=None):
    envs = [make_env(env_name, seed, i) for i in range(num_processes)]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    # Add normalization if needed (common in A2C/PPO)
    # envs = VecNormalize(envs)

    if num_frame_stack is not None:
        envs = VecFrameStack(envs, num_frame_stack)
    elif len(envs.observation_space.shape) == 3:
        # Default to 4 frames for pixel-based environments
        envs = VecFrameStack(envs, 4)

    return envs