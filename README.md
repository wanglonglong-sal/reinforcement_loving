Python: 3.9 (local)
The conda environment in local: conda activate metaDrive_rl
Install deps: pip install -r requirements.txt

- run python command
python -m src.main.2Dworld_SARSA
![2Dworld_SARSA_90_20260116120443](https://github.com/user-attachments/assets/fd906d1b-5eea-4b16-9796-6f73235d066d)


python -m src.main.lineworld
python -m src.main.lineworld_SARSA
![lineworld_SARSA_90_20260116120347](https://github.com/user-attachments/assets/b2b8e28a-f006-45b6-80cc-9f3ce976e788)

Open tensorBoard: tensorboard --logdir runs
http://localhost:6006/
<img width="3071" height="1743" alt="image" src="https://github.com/user-attachments/assets/b4f6e618-20e7-40c3-8607-b94000004b07" />
