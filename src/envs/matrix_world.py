import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config.Config import CONFIG
from src.data.run_context import RunRewards

# Define the physical environment    
class MatrixWorld(gym.Env):
    def __init__(self):
        print("MatrixWorld __int__ called")
        # Define world space size
        self.width = CONFIG["environment"]["width"]
        self.height = CONFIG["environment"]["height"]
        # Define minimum coordinates of the world space
        self.min_x = 0
        self.min_y = 0
        self.min_pos = [self.min_x, self.min_y]
        # Define maximum coordinates of the world space
        self.max_x = self.width - 1
        self.max_y = self.height - 1
        self.max_pos = [self.max_x, self.max_y]
        # Define the start position of the world
        self.start_pos = self.min_pos    
        # Define the goal position of the world
        self.goal_pos = self.max_pos
        # Agent position in the current frame in the world
        self.pos = self.min_pos
        # Agent position in the previous frame in the world
        self.last_pos = self.min_pos
        # Action space: four actions 0:up / 1:down / 2:left / 3:right
        self.action_space = spaces.Discrete(4)
        # Observation space: use MultiDiscrete
        self.observation_space = spaces.MultiDiscrete([self.width, self.height])
        # Define training step constraints
        self.min_step = CONFIG["environment"]["min_steps"]
        self.max_step = CONFIG["environment"]["max_steps"]
        # Initialize rewards
        self.rrwds = None

    def set_rewards(self, rrwds: RunRewards):
        if not isinstance(rrwds, RunRewards):
            raise TypeError("rrwds must be RunRewards")
        self.rrwds = rrwds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Episode start: move the agent back to the start position
        self.pos = self.start_pos.copy()
        # Initialize the agent's previous-frame position
        self.last_pos = tuple(self.start_pos.copy())
        # Randomly set the goal position
        while True:
            gx = self.np_random.integers(self.min_x, self.max_x + 1)
            gy = self.np_random.integers(self.min_y, self.max_y + 1)
            if [gx, gy] != self.pos:
                break
        self.goal_pos = [int(gx), int(gy)]
        # Convert the initial position and goal position into the observation space
        obs = np.concatenate([self.pos, self.goal_pos]).astype(np.int64)
        # Initialize the accumulated training step count
        self.steps = 0
        # Extra info output: not agent-learning related, can be used for debugging
        info = {}
        
        return obs, info

    def step(self, action): #0:up / 1:down / 2:left / 3:right
        # Extract current x, y coordinates
        x, y = self.pos
        before_act_pos = (x ,y)
        # If moving up, y + 1
        if (action == 0):
            y += 1
        # If moving down, y - 1
        elif (action == 1):
            y -= 1
        # If moving left, x - 1
        elif (action == 2):
            x -= 1
        # If moving right, x + 1
        elif (action == 3):
            x += 1
        # Boundary clipping: if current y is out of range, clip back into [0, 4]
        y = int(np.clip(y, self.min_y, self.max_y))
        # Boundary clipping: if current x is out of range, clip back into [0, 4]
        x = int(np.clip(x, self.min_x, self.max_x))
        # Write back the latest position
        self.pos = [x, y]
        after_act_pos = (x, y)
        # Wall-hit check
        hit_wall = False
        if after_act_pos == before_act_pos:
            hit_wall = True
        # Oscillation (back-and-forth) check
        repeat_move = False
        if after_act_pos == self.last_pos:
            repeat_move = True
        # If the goal is reached, mark the episode as terminated and give the maximum reward
        if self.pos == self.goal_pos:
            reward = self.rrwds.goal_pos_reward
            terminated = True
        # If the goal is not reached, the episode continues with penalties    
        else:
            # Regular movement penalty
            reward = self.rrwds.step_reward
            terminated = False
            # Special penalty - wall-hit penalty
            if hit_wall and self.rrwds.hit_wall_enable:
                reward += self.rrwds.hit_wall_reward
            # Special penalty - oscillation (back-and-forth) penalty
            if repeat_move and self.rrwds.repeat_position_enable:
                reward += self.rrwds.repeat_position_reward
        # Convert position info into the observation space
        obs = np.concatenate([self.pos, self.goal_pos]).astype(np.int64)
        # If the step count far exceeds the expectation, truncate the episode
        self.steps += 1 
        if self.steps >= self.max_step:
            truncated = True 
        else:
            truncated = False 
        info = {}
        # Update previous-frame position
        self.last_pos = after_act_pos

        return obs, reward, terminated, truncated, info   
    
# Convert environment observation to state
def obs_to_state(obs, width):
    x, y = obs
    return y * width + x

# Convert state to one-hot
def state_to_onehot(s: int, num_states: int) -> np.ndarray:
    v = np.zeros(num_states, dtype=np.float32)
    v[s] = 1.0
    return v
