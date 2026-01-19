Python: 3.10 (local)

Create conda enviroment
- conda create -n env_rl python=3.10 -y
- conda activate env_rl

Install packages
- pip install -r .\requirements.txt

Install torch, cpu or gpu version
- cpu: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- gpu: conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

Rebuild working environment by env_rl.yml
- conda env create -f xxx_env_rl.yml
- conda activate env_rl


Run python program
- python -m src.main.2Dworld_DQN_ran
- python -m src.main.2Dworld_DQN

Run tensorBoard
- tensorboard --logdir runs
- http:127.0.0.1:10060

Supervise gpu performance
- nvidia-smi -l 1

