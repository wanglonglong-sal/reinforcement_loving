import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("torch cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
import torch.optim as optim
from copy import deepcopy
from config.Config import CONFIG
from src.uti.utilities import get_path_variables
from src.data.run_context import RunContext
from src.envs.matrix_world import MatrixWorld
from src.agents.replay_buffer import ReplayBuffer
from src.agents.dqn_net import QNetRan
from src.trainers.dqn_trainer import train_dqn_ran

# Check whether a GPU is available in the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    # Initialize related path variables
    project_root, execute_filename, execute_stem = get_path_variables()
    rctx = RunContext(
        project_root = project_root,
        execute_file = execute_filename,
        execute_stem = execute_stem    
    )
    # Initialize the environment object
    env = MatrixWorld()
    # Initialize the neural network object
    q_net = QNetRan(6, env.action_space.n).to(device)
    # Create target_net, keeping the structure and parameters consistent with q_net
    target_net = deepcopy(q_net).to(device)
    target_net.load_state_dict(q_net.state_dict())
    # Switch the target network to evaluation mode
    target_net.eval()
    # Initialize the replay buffer object
    replay_buffer = ReplayBuffer()
    # Initialize the optimizer object
    optimizer = optim.Adam(q_net.parameters(), lr=CONFIG["algorithm"]["lr"])

    # Start reinforcement learning training
    train_dqn_ran(env, rctx, q_net, target_net, replay_buffer, optimizer, device)

    print("done")
