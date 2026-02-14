import logging
from typing import Tuple
import numpy as np
import gymnasium as gym # Use gymnasium
import msgpack

# Import the custom Lunar Alighting environment
from .lunar_alighting_env import LunarAlightingEnv

from .envs import make_vec_envs
from .messages import (InfoMessage, MakeMessage, ResetMessage,
                                 StepMessage)
from .zmqServer import ZmqClient

class Server:
    def __init__(self, zmq_client: ZmqClient):
        self.zmq_client: ZmqClient = zmq_client
        self.env = None
        logging.info("Gym server initialized")

    def serve(self):
        logging.info("Serving")
        try:
            self._serve()
        except KeyboardInterrupt:
            logging.info("Server stopped by user")

    def _serve(self):
        while True:
            request = self.zmq_client.receive()
            method = request['method']
            param = request.get('param', {})

            if method == 'info':
                info_data = self.info()
                self.zmq_client.send(InfoMessage(*info_data))

            elif method == 'make':
                try:
                    # Handle C++ field names (envName, numEnv)
                    env_name = param.get('envName') or param.get('env_name')
                    num_envs = param.get('numEnv') or param.get('num_envs')
                    
                    if not env_name or num_envs is None:
                        raise ValueError(f"Missing required parameters. Got: {param}")
                    
                    self.make(env_name, num_envs)
                    self.zmq_client.send(MakeMessage())
                except Exception as e:
                    logging.error("Error in make request: %s", str(e))
                    # Send error response back to client
                    error_response = {"result": f"ERROR: {str(e)}"}
                    self.zmq_client.socket.send(msgpack.packb(error_response))

            elif method == 'reset':
                observation = self.reset()
                self.zmq_client.send(ResetMessage(observation))

            elif method == 'step':
                try:
                    # Handle C++ field names (action, not actions)
                    actions = param.get('action') or param.get('actions')
                    render = param.get('render', False)
                    
                    if actions is None:
                        raise ValueError(f"Missing 'action' parameter. Got: {param}")
                    
                    # Now returning 5 elements from our internal step
                    obs, rew, term, trunc, info = self.step(np.array(actions), render)

                    # Send the updated StepMessage (which handles combining term/trunc)
                    self.zmq_client.send(StepMessage(obs, rew, term, trunc, info['reward']))
                except Exception as e:
                    logging.error("Error in step request: %s", str(e))
                    # Send error response back to client
                    error_response = {"result": f"ERROR: {str(e)}"}
                    self.zmq_client.socket.send(msgpack.packb(error_response))

    def info(self):
        action_space = self.env.action_space
        action_space_type = action_space.__class__.__name__

        # Handle shape for Discrete spaces (which don't have a .shape attribute)
        if hasattr(action_space, 'n'):
            action_space_shape = [action_space.n]
        else:
            action_space_shape = list(action_space.shape)

        observation_space_type = self.env.observation_space.__class__.__name__
        observation_space_shape = list(self.env.observation_space.shape)

        return (action_space_type, action_space_shape,
                observation_space_type, observation_space_shape)

    def make(self, env_name, num_envs):
        logging.info("Making %d %ss", num_envs, env_name)
        try:
            self.env = make_vec_envs(env_name, 0, num_envs)
            logging.info("Successfully created %d %s environments", num_envs, env_name)
        except Exception as e:
            logging.error("Failed to create environment %s: %s", env_name, str(e))
            # Re-raise the exception so the client knows something went wrong
            raise

    def reset(self) -> np.ndarray:
        logging.info("Resetting environments")
        # In Gymnasium, env.reset() returns (obs, info). We just want obs.
        obs = self.env.reset()
        return obs

    def step(self, actions: np.ndarray, render: bool = False):
        # Fix for NumPy 1.24+ (np.int is gone)
        if "Discrete" in str(type(self.env.action_space)):
            actions = actions.squeeze(-1).astype(np.int64)

        # Gymnasium returns 5 values
        observation, reward, terminated, truncated, info = self.env.step(actions)

        # Reshape for your C++ backend expectations
        reward = np.expand_dims(reward, -1)
        terminated = np.expand_dims(terminated, -1)
        truncated = np.expand_dims(truncated, -1)

        if render:
            self.env.render()

        return observation, reward, terminated, truncated, info