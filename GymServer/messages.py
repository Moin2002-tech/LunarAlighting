from abc import ABC, abstractmethod
import numpy as np
import msgpack

class Message(ABC):
    """
    Base class for messages.
    """
    @abstractmethod
    def to_msg(self) -> bytes:
        """
        Creates the MessagePack bytes for the request.
        """

class InfoMessage(Message):
    def __init__(self, action_space_type, action_space_shape,
                 observation_space_type, observation_space_shape):
        self.action_space_type = action_space_type
        self.action_space_shape = action_space_shape
        self.observation_space_type = observation_space_type
        self.observation_space_shape = observation_space_shape

    def to_msg(self) -> bytes:
        request = {
            "actionSpaceType": self.action_space_type,
            "actionSpaceShape": [int(x) for x in self.action_space_shape],
            "observationSpaceType": self.observation_space_type,
            "observationSpaceShape": [int(x) for x in self.observation_space_shape]
        }
        return msgpack.packb(request)

class MakeMessage(Message):
    def to_msg(self) -> bytes:
        return msgpack.packb({"result": "OK"})

class ResetMessage(Message):
    def __init__(self, observation: np.ndarray):
        self.observation = observation

    def to_msg(self) -> bytes:
        # Note: In Gymnasium, reset also returns 'info'.
        # If your C++ side expects it, you might need to add it here.
        # C++ MlpResetResponse expects std::vector<std::vector<float>>
        if self.observation.ndim == 1:
            # Convert 1D array to 2D with one row
            obs_2d = [self.observation.tolist()]
        else:
            # Convert 2D array to list of lists
            obs_2d = self.observation.tolist()
        
        request = {
            "observation": obs_2d
        }
        return msgpack.packb(request)

class StepMessage(Message):
    """
    Updated for Gymnasium compatibility (Python 3.12).
    """
    def __init__(self,
                 observation: np.ndarray,
                 reward: np.ndarray,
                 terminated: np.ndarray,
                 truncated: np.ndarray,
                 real_reward: np.ndarray):
        self.observation = observation
        self.reward = reward
        # We combine terminated and truncated to represent 'done' for legacy C++ code
        self.done = np.logical_or(terminated, truncated)
        self.real_reward = real_reward

    def to_msg(self) -> bytes:
        request = {
            "observation": [[float(x) for x in obs] for obs in self.observation],
            "reward": [[float(x) for x in rew] for rew in self.reward],
            "done": [[bool(x) for x in done] for done in self.done],
            "real_reward": [[float(x) for x in rr] for rr in self.real_reward]
        }
        return msgpack.packb(request)