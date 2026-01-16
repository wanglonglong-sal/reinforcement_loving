import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from src.figures_visualize import draw_learning_curve
from config.Config import CONFIG
from src.utilities import clear_folder

class LineWorld(gym.Env):
    def __init__(self):
        print("LineWorld __int__ called")
        # 动作空间：两个动作 0:left / 1:right
        self.action_space = spaces.Discrete(2)
        # 定义世界空间最小为0
        self.min_pos = 0
        # 定义世界空间最大为9；[0]—[1]—[2]—[3]—[4]—[5]—[6]—[7]—[8]—[9]
        self.max_pos = 9
        # 观测空间：定义智能体可以观测一个连续的一维向量，最小值0.0，最大值1.0，数据为float32位类型
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (1,),
            dtype=np.float32
        )
        # 智能体在世界中的位置状态信息
        self.pos = self.min_pos
        # 定义训练步数约束
        self.min_step = 1
        self.max_step = CONFIG["environment"]["max_steps"]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 一局开始：把智能体放回起点
        self.pos = self.min_pos
        # 将初始化位置转换位观测空间，[self.pos / 9.0] = 归一化数据至[0.0 - 1.0]
        obs = np.array([self.pos / self.max_pos], dtype=np.float32)
        # 初始化训练部署
        self.steps = self.min_step
        # 额外信息反馈出口，非Agent相关学习信息，可用于调试
        info = {}
        
        return obs, info

    def step(self, action):

        # 如果向左，位置减1
        if (action == 0):
            self.pos -= 1
        # 如果向右，位置加1
        elif (action == 1):
            self.pos += 1
        # 边界裁剪，如果当前位置超出0或9，拉回到0-9空间内
        self.pos = int(np.clip(self.pos, self.min_pos, self.max_pos))
        # 抵达终点时标记任务结束，最大化奖励
        if self.pos == self.max_pos:
            reward = 1.0
            terminated = True
        # 未抵达中间时标记任务继续，惩罚    
        else:
            reward = -0.01
            terminated = False
        # 将位置信息归一化到观测空间0-1中表示
        obs = np.array([(self.pos - self.min_pos) / (self.max_pos - self.min_pos)], dtype=np.float32)
        # 当步数远超预期时，中断训练
        if self.steps >= self.max_step: truncated = True 
        else:
            self.steps += 1 
            truncated = False 
        info = {}

        return obs, reward, terminated, truncated, info    

def obs_to_state(obs, max_pos):
    return int(round(obs[0] * max_pos))

def train_episode_initilize(env):
    # 每轮训练初始化观测环境
    obs, info = env.reset()
    # 每轮训练初始化状态
    s = obs_to_state(obs, env.max_pos)
    # 每轮训练初始化结束判断
    done = False
    # 每轮训练初始化步数统计
    step_count = 0

    return obs, info, s, done, step_count    

def evaluate_performance(env, Q):

    obs, info, s, done, eval_steps = train_episode_initilize(env)
    while not done:
        a = int(np.argmax(Q[s]))
        next_obs, reward, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        s_next = obs_to_state(next_obs, env.max_pos)
        s = s_next
        eval_steps += 1    
    return eval_steps

def train_q_learning(env):

    # 计算状态空间
    n_states = env.max_pos - env.min_pos + 1
    # 计算动作空间
    n_actions = env.action_space.n
    # 创建q表 (10,2)
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    # 初始化训练次数
    episodes = CONFIG["training"]["episodes"]
    # 初始化学习率 alpha
    alpha = CONFIG["algorithm"]["alpha"]
    # 初始化折扣因子 gamma，表示未来奖励这算在现在值多少
    gamma = CONFIG["algorithm"]["gamma"]
    # 初始化随机因子相关参数
    epsilon = CONFIG["exploration"]["epsilon_start"]    
    epsilon_decay = CONFIG["exploration"]["epsilon_decay"]
    epsilon_min = CONFIG["exploration"]["epsilon_min"]        
    # 学习效果统计
    episode_steps = []
    # 创建tensorBoard日志，位于runs/
    log_dir=CONFIG["paths"]["log_dir"]
    run_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = Path(log_dir) / run_time
    writer = SummaryWriter(log_dir)

    # 进入训练，训练次数=episodes
    for ep in range(episodes):
        # 每轮训练初始化观测环境
        print("The episode >>>>>>>>> ", ep)
        obs, info, s, done, step_count = train_episode_initilize(env)

        # 开始执行知道抵达终点或任务中断
        while not done:
            if np.random.rand() < epsilon:
                # 随机选择一个动作
                a = env.action_space.sample()
            else:
                # 贪心Q表
                a = int(np.argmax(Q[s]))
            # 执行后得到反馈
            next_obs, reward, terminated, truncated, info = env.step(a)
            # 将观测空间转回状态位置
            s_next = obs_to_state(next_obs, env.max_pos)
            # 是否抵达终点或被打断
            done = terminated or truncated
            # 更新Q表 ★★★★★★
            Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_next]) - Q[s, a])
            # print("count, s, a, Q[s, a]", step_count, s, a, Q[s, a], s_next)
            # 状态推进
            s = s_next
            # 步数累计
            step_count += 1
            if (done): print("This episode is completed.")
        # 随机概率衰减
        if epsilon > epsilon_min:
            epsilon = epsilon * epsilon_decay            
        # 将每轮步数统计到数组    
        episode_steps.append(step_count)
        # 将每轮步数加入tensorBoard
        writer.add_scalar("Episode/Steps", step_count, ep)
        # 将每轮随机率加入tensorBoard
        writer.add_scalar("Episode/Epsilon", epsilon, ep)
        # 评估学习效果，不学习不更新Q表
        eval_steps = evaluate_performance(env, Q)
        writer.add_scalar("Eval/Steps", eval_steps, ep)
    # 画学习率折线图    
    # draw_learning_curve(episode_steps)
    # 关闭tensorBoard文件写入
    writer.close()    

if __name__ == "__main__":
    env = LineWorld()
    train_q_learning(env)
    print("done")
