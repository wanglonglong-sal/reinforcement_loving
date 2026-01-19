import torch
import numpy as np

# Action selection
def epsilon_greedy_dqn(env, epsilon, q_net, s_onehot, device):
    # Random selection
    if np.random.rand() < epsilon:
        # Sample a random action
        return env.action_space.sample()
    # Obtain greedy action via forward inference of the neural network
    with torch.no_grad():
        x = torch.tensor(
            s_onehot, dtype=torch.float32, device=device
        ).unsqueeze(0)
        q_values = q_net(x)
        return int(torch.argmax(q_values, dim=1).item())
    
# Action selection
def epsilon_greedy_dqn_ran(env, epsilon, q_net, obs_np, device):
    # Random selection
    if np.random.rand() < epsilon:
        # Sample a random action
        return env.action_space.sample()
    # Obtain greedy action via forward inference of the neural network
    with torch.no_grad():
        x = torch.from_numpy(obs_np).to(device).unsqueeze(0) 
        q_values = q_net(x)
        return int(torch.argmax(q_values, dim=1).item())
