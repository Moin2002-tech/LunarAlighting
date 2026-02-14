import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class LunarAlightingEnv(gym.Env):
    """
    Custom Lunar Alighting environment that matches the C++ simulation
    """
    metadata = {'render_modes': ['human'], 'render_fps': 60}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Action space: 0=Nothing, 1=Left Engine, 2=Main Engine, 3=Right Engine
        self.action_space = spaces.Discrete(4)
        
        # Observation space: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
        # Normalized to [-1, 1] range like in the C++ code
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(8,), dtype=np.float32
        )
        
        # Physics constants (same as C++)
        self.gravity = 1.62  # Moon gravity
        self.thrust_main = 13.0
        self.thrust_side = 4.0
        self.max_landing_velocity = 2.0
        self.max_angle = 0.2  # radians
        
        # State
        self.state = None
        self.step_count = 0
        self.max_steps = 1000
        
        # Rendering
        self.render_mode = render_mode
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random starting position (similar to C++)
        self.state = np.array([
            random.uniform(-1.0, 1.0),  # x (normalized)
            random.uniform(0.5, 1.0),   # y (normalized, higher up)
            random.uniform(-0.1, 0.1),  # vx
            random.uniform(-0.1, 0.1),  # vy
            random.uniform(-0.1, 0.1),  # angle
            0.0,                        # angular_vel
            0.0,                        # left_leg_contact
            0.0                         # right_leg_contact
        ], dtype=np.float32)
        
        self.step_count = 0
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.state, {}
    
    def step(self, action):
        # Get current state
        x, y, vx, vy, angle, angular_vel, left_contact, right_contact = self.state
        
        # Check if already landed
        if left_contact and right_contact:
            return self.state, 0.0, True, False, {}
        
        # Apply physics (simplified version of C++ physics)
        # Apply gravity
        vy -= self.gravity * 0.01  # dt = 0.01
        
        # Apply thrust based on action
        if action == 1:  # Left engine
            vx += self.thrust_side * np.sin(angle) * 0.01
            vy += self.thrust_side * np.cos(angle) * 0.01
            angular_vel += 0.1 * 0.01
        elif action == 2:  # Main engine
            vx += self.thrust_main * np.sin(angle) * 0.01
            vy += self.thrust_main * np.cos(angle) * 0.01
        elif action == 3:  # Right engine
            vx -= self.thrust_side * np.sin(angle) * 0.01
            vy += self.thrust_side * np.cos(angle) * 0.01
            angular_vel -= 0.1 * 0.01
        
        # Update position and angle
        x += vx * 0.01
        y += vy * 0.01
        angle += angular_vel * 0.01
        
        # Keep angle in range
        angle = np.clip(angle, -np.pi, np.pi)
        
        # Check ground collision (simplified)
        ground_level = 0.0
        if y <= ground_level:
            y = ground_level
            vy = 0
            vx *= 0.8  # friction
            angular_vel *= 0.5
            left_contact = 1.0
            right_contact = 1.0
        
        # Normalize observations
        x_norm = np.clip(x, -1.0, 1.0)
        y_norm = np.clip(y, -1.0, 1.0)
        vx_norm = np.clip(vx / 5.0, -1.0, 1.0)
        vy_norm = np.clip(vy / 5.0, -1.0, 1.0)
        angle_norm = np.clip(angle / np.pi, -1.0, 1.0)
        angular_vel_norm = np.clip(angular_vel / 2.0, -1.0, 1.0)
        
        self.state = np.array([x_norm, y_norm, vx_norm, vy_norm, angle_norm, angular_vel_norm, left_contact, right_contact], dtype=np.float32)
        
        # Calculate reward
        reward = 0.0
        terminated = False
        truncated = False
        
        if left_contact and right_contact:
            # Landed
            if abs(vy) < self.max_landing_velocity and abs(angle) < self.max_angle:
                reward = 100.0  # Successful landing
                terminated = True
            else:
                reward = -100.0  # Crash landing
                terminated = True
        elif y < -0.5:  # Fell off screen
            reward = -100.0
            terminated = True
        else:
            # Small reward for staying alive and moving toward landing
            reward = -0.1  # Small negative reward for each step
            if y > 0:  # Reward for being above ground
                reward += 0.1
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            truncated = True
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.state, reward, terminated, truncated, {}
    
    def _render_frame(self):
        # Simple rendering - just print state for now
        if self.step_count % 50 == 0:  # Print every 50 steps
            print(f"Step {self.step_count}: x={self.state[0]:.2f}, y={self.state[1]:.2f}, vx={self.state[2]:.2f}, vy={self.state[3]:.2f}")
    
    def close(self):
        pass

# Register the environment
gym.register(
    id='LunarAlighting-v1',
    entry_point=LunarAlightingEnv,
    max_episode_steps=1000,
)
