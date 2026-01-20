from src.envs.straight_env import StraightEnv
from src.envs.state import extract_state

def main():
    # 初始化环境信息
    env = StraightEnv(render=True)
    obs, info = env.reset()
    # 定义动作 0.0：steering，转向； 0.2：throttle，油门
    action = [0.0, 0.2]
    # 执行动作得到反馈
    obs2, reward, terminated, truncated, info2 = env.step(action)
    # 更新最新状态
    s1 = extract_state(env.raw())
    # 结束仿真环境
    env.close()

if __name__ == "__main__":
    main()