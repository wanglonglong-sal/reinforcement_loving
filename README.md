Python: 3.10 (local)

Working Environment Installation:
-----------------------------------------------------------------------------------------------------
Create conda enviroment
- conda create -n env_metaDrive python=3.10 -y
- conda activate env_metaDrive

Install torch, cpu or gpu version
- cpu: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
- gpu: conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

Install fundamental packages
- pip install -r .\requirements.txt

Install MetaDrive
- pip install metadrive-simulator

Environment Verification:
- python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('cuda', torch.version.cuda)"
- python -c "import metadrive; print('metadrive ok')"

One-key rebuilt method:
-----------------------------------------------------------------------------------------------------
Rebuild working environment by env_rl.yml
- conda env create -f xxx_env_rl.yml
- conda activate env_rl

Core commands:
-----------------------------------------------------------------------------------------------------
Run python program
- python -m src.main.2Dworl

Run tensorBoard
- tensorboard --logdir runs
- http://localhost:6006/

Supervise gpu performance
- nvidia-smi -l 1

Comments:
-----------------------------------------------------------------------------------------------------
