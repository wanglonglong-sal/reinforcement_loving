import numpy as np

def extract_state(env) -> np.ndarray:
    # 获取当前车辆Agent信息
    agent = env.agent
    # 获取车道线信息
    lane = agent.lane
    # 计算距离车道中心线偏离
    _, lateral = lane.local_coordinates(agent.position)
    # 获得车辆速度信息
    speed = agent.speed_km_h

    return np.array([lateral, speed], dtype=np.float32)