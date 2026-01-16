import numpy as np
import gymnasium as gym
import os
import time
from gymnasium import spaces
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.viz.figures_visualize import draw_learning_curve
from config.Config import CONFIG
from src.utility.utilities import clear_folder, get_path_variables
from src.ani.animation_visualize import animate_position_2d, animate_position_2d_img
from src.data.run_context import RunContext, RunRewards


class MatrixWorld(gym.Env):
    def __init__(self):
        print("MatrixWorld __int__ called")

        # Define the environment size
        self.width = CONFIG["environment"]["width"]
        self.height = CONFIG["environment"]["height"]

        # Define the minimum coordinate of the environment
        self.min_x = 0
        self.min_y = 0
        self.min_pos = [self.min_x, self.min_y]

        # Define the maximum coordinate of the environment
        self.max_x = self.width - 1
        self.max_y = self.height - 1
        self.max_pos = [self.max_x, self.max_y]

        # Agent position/state in the environment
        self.pos = self.min_pos

        # Action space: 4 actions 0: up / 1: down / 2: left / 3: right
        self.action_space = spaces.Discrete(4)

        # Observation space: use multi-discrete space
        self.observation_space = spaces.MultiDiscrete([self.width, self.height])

        # Define step constraints
        self.min_step = CONFIG["environment"]["min_steps"]
        self.max_step = CONFIG["environment"]["max_steps"]

        # Reward configuration (injected later)
        self.rrwds = None

    def set_rewards(self, rrwds: RunRewards):
        if not isinstance(rrwds, RunRewards):
            raise TypeError("rrwds must be RunRewards")
        self.rrwds = rrwds

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # At the start of an episode: reset the agent to the start position
        self.pos = self.min_pos.copy()

        # Convert initial position into observation space format
        obs = np.array(self.pos, dtype=np.int64)

        # Reset step counter for the episode
        self.steps = 0

        # Additional info (not used by the agent), useful for debugging
        info = {}

        return obs, info

    def step(self, action):  # 0: up / 1: down / 2: left / 3: right
        # Extract current (x, y)
        x, y = self.pos
        old_x = x
        old_y = y

        # If action is up, y + 1
        if action == 0:
            y += 1
        # If action is down, y - 1
        elif action == 1:
            y -= 1
        # If action is left, x - 1
        elif action == 2:
            x -= 1
        # If action is right, x + 1
        elif action == 3:
            x += 1

        # Boundary clamp for y
        y = int(np.clip(y, self.min_y, self.max_y))
        # Boundary clamp for x
        x = int(np.clip(x, self.min_x, self.max_x))

        # Update agent position
        self.pos = [x, y]

        # Detect wall hit (no movement after clamping)
        hit_wall = False
        if x == old_x and y == old_y:
            hit_wall = True

        # If reaching the goal: terminate and grant goal reward
        if self.pos == self.max_pos:
            reward = self.rrwds.max_pos_reward
            terminated = True
        # Otherwise: continue with step penalty (and optional wall penalty)
        else:
            reward = self.rrwds.step_reward
            terminated = False
            if hit_wall and self.rrwds.hit_wall_enable:
                reward += self.rrwds.hit_wall_reward

        # Convert position into observation format
        obs = np.array(self.pos, dtype=np.int64)

        # Truncate if the episode exceeds max steps
        self.steps += 1
        if self.steps >= self.max_step:
            truncated = True
        else:
            truncated = False

        info = {}

        return obs, reward, terminated, truncated, info

    def render(self, delay=0.08, clear=True):
        if clear:
            os.system("cls" if os.name == "nt" else "clear")
        cells = ["-"] * (self.max_pos - self.min_pos + 1)
        cells[self.max_pos] = "G"
        cells[self.pos] = "♥"
        print("".join(cells))

        time.sleep(delay)


def obs_to_state(obs, width):
    x, y = obs
    return y * width + x


def train_episode_initilize(env):
    # Reset environment at the beginning of each episode
    obs, info = env.reset()

    # Convert observation to a discrete state index
    s = obs_to_state(obs, env.width)

    # Episode termination flag
    done = False

    # Step counter within the episode
    step_count = 0

    return obs, info, s, done, step_count


def evaluate_performance(env, Q):
    # Initialize environment and variables for evaluation
    obs, info, s, done, eval_steps = train_episode_initilize(env)

    # Record positions
    x, y = obs
    positions = []
    positions.append((int(x), int(y)))

    # Record actions
    actions = []

    while not done:
        # Select the greedy action
        a = int(np.argmax(Q[s]))
        actions.append(a)

        # Step in the environment
        next_obs, reward, terminated, truncated, info = env.step(a)

        # Check termination/truncation
        done = terminated or truncated

        # Convert next observation to next state index
        s_next = obs_to_state(next_obs, env.width)

        # Move forward
        s = s_next
        obs = next_obs

        # Count steps
        eval_steps += 1

        # Store the next position
        x, y = next_obs
        positions.append((int(x), int(y)))

    return eval_steps, positions, actions


def epsilon_greedy(epsilon, env, Q, s):
    if np.random.rand() < epsilon:
        # Sample a random action
        a = env.action_space.sample()
    else:
        # Greedy action from Q-table
        a = int(np.argmax(Q[s]))

    return a


def train_sarsa(env, rctx):
    # Compute state space size
    n_states = env.width * env.height

    # Compute action space size
    n_actions = env.action_space.n

    # Create Q-table (n_states, n_actions)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    # Training episodes
    episodes = CONFIG["training"]["episodes"]

    # Learning rate alpha
    alpha = CONFIG["algorithm"]["alpha"]

    # Discount factor gamma (how much future rewards are worth now)
    gamma = CONFIG["algorithm"]["gamma"]

    # Epsilon-greedy exploration parameters
    epsilon = CONFIG["exploration"]["epsilon_start"]
    epsilon_decay = CONFIG["exploration"]["epsilon_decay"]
    epsilon_min = CONFIG["exploration"]["epsilon_min"]

    # TensorBoard log directory (under runs/)
    log_dir = CONFIG["paths"]["log_dir"]
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = Path(log_dir) / f"{rctx.execute_stem}_{run_time}"
    writer = SummaryWriter(log_dir)

    # Animation settings
    save_gif = CONFIG["animation"]["save_gif"]
    save_gif_ep = CONFIG["animation"]["save_gif_ep"]
    save_gif_ep_start = CONFIG["animation"]["save_git_ep_start"]
    agent_img_dir = CONFIG["animation"]["agent_img_dir"]
    des_img_dir = CONFIG["animation"]["des_img_dir"]
    end_img_path = CONFIG["animation"]["end_img_path"]
    ending_time = CONFIG["animation"]["ending_time"]

    # Reward settings
    max_pos_reward = CONFIG["rewards"]["max_pos_reward"]
    step_reward = CONFIG["rewards"]["step_reward"]
    hit_wall_enable = CONFIG["rewards"]["hit_wall_enable"]
    hit_wall_reward = CONFIG["rewards"]["hit_wall_reward"]

    rrwds = RunRewards(
        max_pos_reward=max_pos_reward,
        step_reward=step_reward,
        hit_wall_enable=hit_wall_enable,
        hit_wall_reward=hit_wall_reward
    )
    env.set_rewards(rrwds)

    # Training loop
    for ep in range(episodes):
        # Initialize environment for each episode
        print("The episode >>>>>>>>> ", ep)
        obs, info, s, done, step_count = train_episode_initilize(env)

        # Select the first action
        a = epsilon_greedy(epsilon, env, Q, s)

        # Rollout until termination or truncation
        while not done:
            # env.render()

            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(a)

            # Convert observation to next state
            s_next = obs_to_state(next_obs, env.width)

            # Check if episode ends
            done = terminated or truncated

            # SARSA update
            if done:
                # If terminal, target is just the immediate reward
                td_target = reward
                a_next = 0
            else:
                # Otherwise, bootstrap with next state-action value
                a_next = epsilon_greedy(epsilon, env, Q, s_next)
                td_target = reward + gamma * Q[s_next, a_next]

            Q[s, a] = Q[s, a] + alpha * (td_target - Q[s, a])

            # Advance state and action
            s = s_next
            a = a_next

            # Count steps
            step_count += 1
            if done:
                print("This episode is completed.")

        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay

        # Log training stats
        writer.add_scalar("Episode/Steps", step_count, ep)
        writer.add_scalar("Episode/Epsilon", epsilon, ep)

        # Evaluate policy (no learning / no Q update)
        eval_steps, positions, actions = evaluate_performance(env, Q)
        writer.add_scalar("Eval/Steps", eval_steps, ep)

        # Generate animation if conditions are met
        if save_gif and ep >= save_gif_ep_start and ep % save_gif_ep == 0:
            run_time = datetime.now().strftime("%Y%m%d%H%M%S")
            ani_path = log_dir / f"{rctx.execute_stem}_{ep}_{run_time}.gif"
            # ani = animate_position_2d(env, positions, actions, ani_path)  # Non-custom animation
            ani = animate_position_2d_img(
                env, positions, actions, ani_path,
                agent_img_dir, des_img_dir, end_img_path,
                terminated, ending_time
            )  # Custom animation version

    # Plot learning curve (if enabled)
    # draw_learning_curve(episode_steps)

    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    # Initialize common path variables
    project_root, execute_filename, execute_stem = get_path_variables()
    rctx = RunContext(
        project_root=project_root,
        execute_file=execute_filename,
        execute_stem=execute_stem
    )

    # Initialize environment
    env = MatrixWorld()

    # Start reinforcement learning training
    train_sarsa(env, rctx)

    print("done")
