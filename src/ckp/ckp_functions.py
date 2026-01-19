import os
import torch

# Save DQN network parameters
def save_dqn_ckpt(path, q_net, target_net, optimizer, episode, epsilon, global_step):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt={
        "q_net":q_net.state_dict(),
        "target_net": target_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": int(episode),                                             
        "epsilon": float(epsilon),
        "global_step": int(global_step)
    }

    torch.save(ckpt, path)

# Reload DQN network parameters to resume training
def load_dqn_ckpt_resume(path, q_net, target_net, optimizer, device):
    
    ckpt = torch.load(path, map_location=device)
    q_net.load_state_dict(ckpt["q_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_episode = ckpt["optimizer"] + 1
    epsilon = ckpt.get("epsilon", 1.0)
    global_step = ckpt["global_step"] + 1

    return start_episode, epsilon, global_step

# Reload DQN network parameters for training in a new scenario
def load_dqn_ckpt_transfer(path, q_net, target_net, device):
    
    ckpt = torch.load(path, map_location=device)
    q_net.load_state_dict(ckpt["q_net"])
    target_net.load_state_dict(ckpt["target_net"])
