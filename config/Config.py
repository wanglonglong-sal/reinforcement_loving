CONFIG = {
    "training":{
        "episodes":1000,              # Number of training episodes 
        "train_record_fre":1000,      # Training data logging frequency, based on global steps
        "eval_performance_fre": 20,  # Performance evaluation frequency, based on episode count
        "train_mode":0,               # 0: training from scratch; 1: resume training under the same observation environment; 2: resume training under a different observation environment
        "load_resume_file_path":"ckps\\best.pt",  # Path to previously saved parameters to load when train mode is 1 or 2
        "pt_save_enabled":False       # enable the function of save pt file      
    },
    "algorithm":{
        "alpha" : 0.1,   # Learning rate
        "gamma" : 0.99,  # Discount factor for future rewards
        "batch_size": 32,                   # Batch size for samples - used in DQN   
        "lr": 1e-3,                         # Neural network learning rate - DQN
        "target_update_freq": 200           # Target network parameter update frequency - DQN
    },
    "exploration":{
        "epsilon_start":0.3,    # Initial exploration rate
        "epsilon_decay":0.9,    # Decay rate
        "epsilon_min":0.01      # Minimum exploration rate

    },
    "environment":{
        "width":5,       # Width of the 2D space 
        "height":5,      # Height of the 2D space
        "min_steps":1,    # Minimum number of steps
        "max_steps":500  # Maximum number of steps
    },
    "paths":{
        "log_dir":"runs",    # Log directory
        "ckp_dir":"ckps",    # Checkpoint directory
        "ani_dir":"anis"     # Animation directory - not used
    },
    "animation":{
        "save_gif":True,                # Whether to save animation
        "save_git_ep_start":800,        # Starting episode for saving animations
        "save_gif_ep":45,               # Episode interval for saving animations 
        "agent_img_dir":"imgs\\G.gif",  # Agent image path
        "des_img_dir":"imgs\\T.gif",    # Target image path    
        "end_img_path":"imgs\\Ending.gif", # Victory screen image path   
        "ending_time":10                # Duration of the victory screen  

    },
    "rewards":{
        "goal_pos_reward":10,       # Reward for reaching the goal
        "step_reward":-0.1,       # Penalty per step
        "hit_wall_enable":True,    # Enable wall-collision penalty
        "hit_wall_reward":-1,    # Wall-collision penalty
        "repeat_position_enable":True,  # Enable oscillation (back-and-forth) penalty
        "repeat_position_reward":-0.5   # Oscillation (back-and-forth) penalty
    }
}
