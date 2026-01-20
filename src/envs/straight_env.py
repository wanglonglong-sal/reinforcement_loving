from metadrive.envs.metadrive_env import MetaDriveEnv
from config.straight_config import straight_config

# 直线环境信息设定
class StraightEnv:
    def __init__(self, render):
        self._env = MetaDriveEnv(straight_config(render=render))

    def reset(self):
        return self._env.reset()
    
    def step(self, action):
        return self._env.step(action)
    
    def close(self):
        self._env.close()

    def raw(self):
        return self._env