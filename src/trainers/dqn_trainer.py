import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from config.Config import CONFIG
from src.data.run_context import RunRewards
from src.agents.policies import epsilon_greedy_dqn_ran
from src.ani.animation_visualize import animate_position_2d_img_ran
from src.ckp.ckp_functions import save_dqn_ckpt, load_dqn_ckpt_resume, load_dqn_ckpt_transfer

# Convert obs to model input features
def obs_to_feature(obs, env):
    # Convert current coordinates and random goal coordinates
    x, y, gx, gy = obs
    # Compute relative distance and normalize
    dx = (gx - x) / env.max_x
    dy = (gy - y) / env.max_y
    # Return the data format that can be fed into the model

    return np.array([x, y, gx, gy, dx, dy], dtype=np.float32)


# Initialize per-episode training state
def train_episode_initilize(env):
    # Reset observation environment for each episode
    obs, info = env.reset()
    # Initialize done flag for each episode
    done = False
    # Initialize step counter for each episode
    step_count = 0

    return obs, info, done, step_count    

def evaluate_performance(env, epsilon, q_net, n_states, device):
    # Initialize environment and variables for each evaluation run
    obs, info, done, eval_steps = train_episode_initilize(env)
    # Update positions record
    x, y, gx, gy = obs
    positions = []
    positions.append((int(x), int(y)))
    # Update actions record
    actions = []
    while not done:
        # Obtain greedy action via forward inference of the neural network
        obs_np = obs_to_feature(obs, env)
        a = epsilon_greedy_dqn_ran(env, epsilon, q_net, obs_np, device) 
        # Append chosen action
        actions.append(a)            
        # Execute action and get feedback
        next_obs, reward, terminated, truncated, info = env.step(a)
        # Check whether goal is reached or timed out
        done = terminated or truncated
        # Save next-step coordinates into positions
        x, y, gx, gy = next_obs
        positions.append((int(x), int(y)))  
        # Accumulate step count
        eval_steps += 1              
        # Advance state
        obs = next_obs
        
    return eval_steps, positions, actions, terminated


def train_dqn_ran(env, rctx, q_net, target_net, replay_buffer, optimizer, device):

    # Compute state space size
    n_states = env.width * env.height
    # Compute action space size
    n_actions = env.action_space.n
    # Initialize number of training episodes
    episodes = CONFIG["training"]["episodes"]
    # Initialize training data logging frequency
    train_record_fre = CONFIG["training"]["train_record_fre"]
    # Initialize performance evaluation frequency
    eval_performance_fre = CONFIG["training"]["eval_performance_fre"]
    # Initialize training mode
    train_mode = CONFIG["training"]["train_mode"]
    # Initialize checkpoint path for resuming training
    load_resume_file_path = CONFIG["training"]["load_resume_file_path"]
    pt_save_enabled = CONFIG["training"]["pt_save_enabled"]
    # Initialize learning rate alpha
    alpha = CONFIG["algorithm"]["alpha"]
    # Initialize discount factor gamma (how much future rewards are worth now)
    gamma = CONFIG["algorithm"]["gamma"]
    # Initialize DQN learning settings
    batch_size = CONFIG["algorithm"]["batch_size"]
    # Initialize DQN target network update frequency
    target_update_freq = CONFIG["algorithm"]["target_update_freq"]
    # Initialize epsilon-related parameters
    epsilon = CONFIG["exploration"]["epsilon_start"]    
    epsilon_decay = CONFIG["exploration"]["epsilon_decay"]
    epsilon_min = CONFIG["exploration"]["epsilon_min"]        
    # Learning statistics
    # episode_steps = []
    # Create tensorBoard logs under runs/
    log_dir=CONFIG["paths"]["log_dir"]
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = Path(log_dir) / f"{rctx.execute_stem}_{run_time}"
    writer = SummaryWriter(log_dir)
    # Set checkpoint save directory
    ckp_dir = CONFIG["paths"]["ckp_dir"]
    # Initialize animation settings
    save_gif = CONFIG["animation"]["save_gif"]
    save_gif_ep = CONFIG["animation"]["save_gif_ep"]
    save_gif_ep_start = CONFIG["animation"]["save_git_ep_start"]
    agent_img_dir = CONFIG["animation"]["agent_img_dir"]
    des_img_dir = CONFIG["animation"]["des_img_dir"]
    end_img_path = CONFIG["animation"]["end_img_path"]
    ending_time = CONFIG["animation"]["ending_time"]
    # Initialize reward settings
    goal_pos_reward = CONFIG["rewards"]["goal_pos_reward"]
    step_reward = CONFIG["rewards"]["step_reward"]
    hit_wall_enable = CONFIG["rewards"]["hit_wall_enable"]
    hit_wall_reward = CONFIG["rewards"]["hit_wall_reward"]
    repeat_position_enable = CONFIG["rewards"]["repeat_position_enable"]
    repeat_position_reward = CONFIG["rewards"]["repeat_position_reward"]
    rrwds = RunRewards(
        goal_pos_reward = goal_pos_reward,
        step_reward = step_reward,
        hit_wall_enable = hit_wall_enable,
        hit_wall_reward = hit_wall_reward,
        repeat_position_enable = repeat_position_enable,
        repeat_position_reward = repeat_position_reward
    )
    env.set_rewards(rrwds)
    # Global step counter (not reset)
    global_step = 0
    # Global timer (not reset)
    t0 = time.perf_counter()
    # If training mode is "resume on the same map", load checkpoint
    if train_mode == 1:
        start_episode, epsilon, global_step = load_dqn_ckpt_resume(load_resume_file_path, q_net, target_net, optimizer, device)
    elif train_mode == 2:
        load_dqn_ckpt_transfer(load_resume_file_path, q_net, target_net, device)
    # Start training, number of episodes = episodes
    for ep in range(episodes):
        # Restore variables for resume-training mode
        if train_mode == 1:
            ep = start_episode
        # Reset observation environment for each episode    
        print("The episode >>>>>>>>> ", ep)
        obs, info, done, step_count = train_episode_initilize(env)
        # Define network learning counter
        q_net_learn_count = 0
        # Run until reaching the goal or being interrupted
        while not done:
            # Select an action
            obs_np = obs_to_feature(obs, env)
            a = epsilon_greedy_dqn_ran(env, epsilon, q_net, obs_np, device) 
            # Execute action and get feedback
            next_obs, reward, terminated, truncated, info = env.step(a)
            next_obs_np = obs_to_feature(next_obs, env)
            # # Record previous-frame position
            # last_x, last_y, last_gx, last_gy = obs
            # env.last_pos = [last_x, last_y]
            # Check whether goal is reached or the episode is truncated
            done = terminated or truncated
            # Add this transition to the replay buffer
            replay_buffer.add(obs_np, a, reward, next_obs_np, done)
            # Trigger neural network learning when enough samples are available
            if len(replay_buffer) >= batch_size:
                # Sample a batch from the replay buffer
                batch = replay_buffer.sample(batch_size)
                # Unpack sampled data into current features, actions, rewards, next features, done flags
                Fea, A, R, Fea2, D = zip(*batch)
                # Convert {current observation} into a tensor-ready format
                Fea_np = np.stack(Fea, axis=0).astype(np.float32)
                Fea = torch.from_numpy(Fea_np).to(device)
                # Convert {next observation} into a tensor-ready format 
                Fea2_np = np.stack(Fea2, axis=0).astype(np.float32)
                Fea2 = torch.from_numpy(Fea2_np).to(device)                   
                # Convert {actions} into a tensor-ready format
                A = torch.tensor(A, dtype=torch.int64, device=device).unsqueeze(1)                
                # Convert {rewards} into a tensor-ready format
                R = torch.tensor(R, dtype=torch.float32, device=device)
                # Convert {done flags} into a tensor-ready format
                D = torch.tensor(D, dtype=torch.float32, device=device)
                # Forward pass on current state to get all Q-values, then gather the Q-value for the executed action
                q_sa = q_net(Fea).gather(1, A).squeeze(1)
                # Forward pass on next state to compute the target value
                with torch.no_grad():
                    # target_next_max = target_net(Fea2).max(dim=1).values
                    q_next_action = q_net(Fea2).argmax(dim=1)
                    target_next_max = target_net(Fea2).gather(1, q_next_action.unsqueeze(1)).squeeze(1)
                    y = R + gamma * (1.0 - D) * target_next_max
                # Compute loss by comparing current Q-values with targets
                loss = F.smooth_l1_loss(q_sa, y)
                # Clear gradients
                optimizer.zero_grad()
                # Backpropagation
                loss.backward()
                # Update parameters
                optimizer.step()
                # Update counter
                q_net_learn_count += 1
                # Update target network parameters
                if q_net_learn_count > target_update_freq:
                    target_net.load_state_dict(q_net.state_dict())
                    q_net_learn_count = 0
                if global_step % train_record_fre == 0:
                # Log TD Error to tensorBoard by step
                    td_error = torch.abs(q_sa - y).mean().item()
                    writer.add_scalar("Train/TD_Error", td_error, global_step)
                    writer.add_scalar("Train/Q_mean", q_sa.mean().item(), global_step)
                    writer.add_scalar("Train/Q_max", q_sa.max().item(), global_step)
                    writer.add_scalar("Train/Loss", loss.item(), global_step)
                    # Training time statistics
                    t1 = time.perf_counter()
                    dt = t1 - t0
                    sec_per_stage = dt
                    sps = train_record_fre / dt if dt > 0 else 0.0
                    writer.add_scalar("Train/sec_per_stage", sec_per_stage, global_step)
                    writer.add_scalar("Train/sps", sps, global_step)
                    t0 = t1

            # Advance state
            obs = next_obs
            # Accumulate step count
            global_step += 1
            step_count += 1
            if (done): print("This episode is completed.")

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay            
        # Log per-episode step count to tensorBoard
        writer.add_scalar("Episode/Steps", step_count, ep)
        # Log per-episode epsilon to tensorBoard
        writer.add_scalar("Episode/Epsilon", epsilon, ep)
        if ep % eval_performance_fre == 0:
            # Evaluate performance (no learning, no Q-table updates)
            eval_steps, positions, actions, eval_terminated = evaluate_performance(env, epsilon, q_net, n_states, device)
            # Log evaluation results to tensorBoard
            writer.add_scalar("Eval/Steps", eval_steps, ep)
            writer.add_scalar("Eval/Success", eval_terminated, ep)
            # Save intermediate checkpoints
            if pt_save_enabled and ep % (eval_performance_fre * 2) == 0:
                run_time = datetime.now().strftime("%Y%m%d%H%M%S")
                ckp_path = Path(ckp_dir) / f"{rctx.execute_stem}_{ep}_{run_time}.pt"
                save_dqn_ckpt(ckp_path, q_net, target_net, optimizer, ep, epsilon, global_step)
            # Generate animation when conditions are met
            if save_gif and ep >= save_gif_ep_start: 
                run_time = datetime.now().strftime("%Y%m%d%H%M%S")
                ani_path = log_dir / f"{rctx.execute_stem}_{ep}_{run_time}.gif"
                # ani = animate_position_2d(env, positions, actions, ani_path)  # Non-custom animation
                ani = animate_position_2d_img_ran(env, positions, actions, ani_path, agent_img_dir, des_img_dir, end_img_path, terminated, ending_time) # Customized animation version
            
    # Close tensorBoard writer
    writer.close()
