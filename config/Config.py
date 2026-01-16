CONFIG = {
    "training": {
        "episodes": 100  # Number of training episodes
    },
    "algorithm": {
        "alpha": 0.1,   # Learning rate
        "gamma": 0.99   # Discount factor for future rewards
    },
    "exploration": {
        "epsilon_start": 0.3,  # Initial exploration rate (epsilon)
        "epsilon_decay": 0.9,  # Epsilon decay rate
        "epsilon_min": 0.01    # Minimum exploration rate
    },
    "environment": {
        "width": 10,        # Width of the 2D environment
        "height": 5,        # Height of the 2D environment
        "min_steps": 1,     # Minimum number of steps per episode
        "max_steps": 1000   # Maximum number of steps per episode
    },
    "paths": {
        "log_dir": "runs",  # Directory for TensorBoard logs
        "ani_dir": "anis"  # Directory for saved animations
    },
    "animation": {
        "save_gif": True,             # Whether to save animation as GIF
        "save_git_ep_start": 90,      # Episode index to start saving GIFs
        "save_gif_ep": 10,            # Interval (episodes) between saved GIFs
        "agent_img_dir": "imgs\\G.gif",       # Path to agent sprite GIF
        "des_img_dir": "imgs\\T.gif",         # Path to target sprite GIF
        "end_img_path": "imgs\\Ending.gif",   # Path to victory animation GIF
        "ending_time": 10              # Duration of victory animation (frames)
    },
    "rewards": {
        "max_pos_reward": 1,           # Reward for reaching the target position
        "step_reward": -0.01,          # Step penalty per action
        "hit_wall_enable": True,       # Enable wall collision penalty
        "hit_wall_reward": -0.5        # Penalty for hitting a wall
    }
}
