import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv

# 设定metaDrive初始化环境
# - "map": "S"，S=Straight, 将地图初始化成一条直行路线
# - "traffic_density": 0.0，无其他交通参与者
# - "use_render": True，允许视频弹出
env = MetaDriveEnv({"map": "S", "traffic_density": 0.0, "use_render": True})

# 初始化环境得到环境观测信息
obs, info = env.reset()

# 获取车辆Agent信息
agent = env.agent  

# 获取车辆可见的车道信息，其中包括 车道中心线的几何形状，车道方向，车道宽度，车道坐标系，lane的参考线
lane = agent.lane

# 计算车辆与车道产生的横纵向偏移，longi(s)：沿着车道前进方向走了多远； lateral(l)：相对车道中心线的横向偏移
longi, lateral = lane.local_coordinates(agent.position)

print("longi:", longi)
print("lateral (offset to lane center):", lateral)
print("speed:", agent.speed_km_h)

# 定义动作 0.0：steering，转向； 0.2：throttle，油门
action = [0.0, 0.2]
# 执行动作得到反馈
obs2, reward, terminated, truncated, info2 = env.step(action)

# 重新计算与车道产生的横纵向偏移
longi2, lateral2 = lane.local_coordinates(agent.position)
print("longi:", longi2)
print("lateral (offset to lane center):", lateral2)
print("speed:", agent.speed_km_h)

# 定义Agent状态
state = np.array(
    [
        lateral, # 横向偏移
        agent.speed_km_h # 速度
    ],
    dtype=np.float32
)

print("state:", state, "shape", state.shape)

def get_state(env):
    agent = env.agent
    lane = agent.lane

    _, lateral = lane.local_coordinates(agent.position)
    speed = agent.speed_km_h

    state = np.array([lateral, speed], dtype=np.float32)
    return state

env.close()
print("done")