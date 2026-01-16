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
from src.ani.animation_visualize import animate_position_1d
from src.data.run_context import RunContext


class LineWorld(gym.Env):
    def __init__(self):
        print("LineWorld __int__ called")

        # Action space: 2 actions 0: left / 1: right
        self.action_space = spaces.Discrete(2)

        # Define the minimum position as 0
        self.min_pos = 0

        # Define the maximum position as 9; [0]-[1]-[2]-[3]-[4]-[5]-[6]-[7]-[8]-[9]
        self.max_pos = 9

        # Observation space: a 1D continuous vector in [0.0, 1.0], dtype=float32
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Agent position/state in the environment
        self.pos = self.min_pos

        # Step constraints for training
        self.min_step = 1
        self.max_step = CONFIG["environment"]["max_steps"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # At the start of an episode: reset the agent to the start position
        self.pos = self.min_pos

        # Convert initial position into observation space (normalized to [0.0, 1.0])
        obs = np.array([self.pos / self.max_pos], dtype=np.float32)

        # Initialize step counter
        self.steps = self.min_step

        # Additional info (not used by the agent), useful for debugging
        info = {}

        return obs, info

    def step(self, action):
        # If action is left, position - 1
        if action == 0:
            self.pos -= 1
        # If action is right, position + 1
        elif action == 1:
            self.pos += 1

        # Boundary clamp: keep position within [0, 9]
        self.pos = int(np.clip(self.pos, self.min_pos, self.max_pos))

        # If reaching the goal: terminate and grant goal reward
        if self.pos == self.max_pos:
            reward = 1.0
            terminated = True
        # Otherwise: continue with step penalty
        else:
            reward = -0.01
            terminated = False

        # Normalize position to observation space [0, 1]
        obs = np.array(
            [(self.pos - self.min_pos) / (self.max_pos - self.min_pos)],
            dtype=np.float32
        )

        # Truncate if the episode exceeds max steps
        if self.steps >= self.max_step:
            truncated = True
        else:
            self.steps += 1
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


def obs_to_state(obs, max_pos):
    return int(round(obs[0] * max_pos))


def train_episode_initilize(env):
    # Reset environment at the beginning of each episode
    obs, info = env.reset()

    # Convert observation to a discrete state index
    s = obs_to_state(obs, env.max_pos)

    # Episode termination flag
    done = False

    # Step counter within the episode
    step_count = 0

    return obs, info, s, done, step_count


def evaluate_performance(env, Q):
    obs, info, s, done, eval_steps = train_episode_initilize(env)
    positions = [env.pos]

    while not done:
        a = int(np.argmax(Q[s]))
        next_obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        s_next = obs_to_state(next_obs, env.max_pos)
        s = s_next
        eval_steps += 1
        positions.append(env.pos)

    return eval_steps, positions


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
    n_states = env.max_pos - env.min_pos + 1

    # Compute action space size
    n_actions = env.action_space.n

    # Create Q-table (10, 2)
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

    # Training statistics
    episode_steps = []

    # TensorBoard log directory (under runs/)
    log_dir = CONFIG["paths"]["log_dir"]
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = Path(log_dir) / f"{rctx.execute_stem}_{run_time}"
    writer = SummaryWriter(log_dir)

    # Animation settings
    save_gif = CONFIG["animation"]["save_gif"]
    save_gif_ep = CONFIG["animation"]["save_gif_ep"]

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

            # Step environment and get feedback
            next_obs, reward, terminated, truncated, info = env.step(a)

            # Convert observation to next state
            s_next = obs_to_state(next_obs, env.max_pos)

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

        # Record steps per episode
        episode_steps.append(step_count)

        # Log stats to TensorBoard
        writer.add_scalar("Episode/Steps", step_count, ep)
        writer.add_scalar("Episode/Epsilon", epsilon, ep)

        # Evaluate policy (no learning / no Q update)
        eval_steps, positions = evaluate_performance(env, Q)
        writer.add_scalar("Eval/Steps", eval_steps, ep)

        # Save animation periodically
        if save_gif and ep % save_gif_ep == 0:
            run_time = datetime.now().strftime("%Y%m%d%H%M%S")
            ani_path = log_dir / f"{rctx.execute_stem}_{ep}_{run_time}.gif"
            ani = animate_position_1d(positions, env.max_pos, ani_path)

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
    env = LineWorld()

    # Start reinforcement learning training
    train_sarsa(env, rctx)

    print("done")
