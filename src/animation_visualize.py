import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from PIL import Image
import numpy as np

ACTION_NAME = {
    0:"UP",
    1:"DOWN",
    2:"LEFT",
    3:"RIGHT"
}

ACTION_ARROW = {
    0:"^",
    1:"v",
    2:"<",
    3:">",
    9:"",
}

def animate_position_1d(positions, max_pos, gif_path):
    # 初始化画布
    fig, ax = plt.subplots()

    ax.set_xlim(-0.5, max_pos + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks(range(max_pos + 1))
    ax.set_title("LineWorld rollout")
    # 设置终点位置
    ax.scatter([max_pos], [0], marker="*", s=250)
    # 设置物体/智能体
    agent_dot, = ax.plot([positions[0]], [0], marker="o", markersize=14)

    def update(frame_idx):
        x = positions[frame_idx]
        agent_dot.set_data([x], [0])
        return agent_dot,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval = 150,
        blit=False,
        repeat=False
    )

    # plt.show()
    ani.save(gif_path, writer="pillow")
    return ani

def animate_position_2d(env, positions, actions, gif_path):
    # 初始化画布
    fig, ax = plt.subplots()

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    # 刻度
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    # 隐藏坐标轴数字，只保留格子
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # 打开网格
    ax.grid(True)
    # 方格等长
    ax.set_aspect('equal')
    # 图形表头
    ax.set_title(f"2D {env.width}*{env.height} Matrix rollout")
    # 设置终点位置
    max_x, max_y = env.max_pos
    ax.scatter(max_x + 0.5, max_y + 0.5, marker="*", s=250)
    # 设置物体/智能体显示
    x, y = positions[0]
    agent_dot, = ax.plot(x + 0.5, y + 0.5, marker="o", markersize=14)
    # 设置物体动作
    arrow = ACTION_ARROW.get(actions[1], "?")
    agent_action, = ax.plot(x + 0.5, y + 0.5, marker=arrow, markersize=10)
    # 计步器
    step_test = ax.text(
        0.02, 0.98, "Step: 0",
        transform=ax.transAxes,
        va="top", ha="left"
    )

    def update(frame_idx):
        # 更新物体位置显示
        x, y = positions[frame_idx]
        agent_dot.set_data([x + 0.5], [y + 0.5])
        # 更新动作显示
        if frame_idx == 0:
            a = 9
        else:
            a = actions[frame_idx - 1]
        arrow = ACTION_ARROW.get(a, "?")
        agent_action.set_data([x + 0.5], [y + 0.5])
        agent_action.set_marker(arrow)
        # 更新步数显示
        step_test.set_text(f"Step: {frame_idx}")

        return agent_dot, step_test

    ani = FuncAnimation(
        fig,
        update,
        frames=len(positions),
        interval = 500,
        blit=False,
        repeat=False
    )

    # plt.show()
    ani.save(gif_path, writer="pillow")
    return ani

def animate_position_2d_img(env, positions, actions, gif_path, agent_img_path, destination_img_path, ending_img_path, terminated, ending_time):
    # 初始化画布
    fig, ax = plt.subplots()
    # 宽，高
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    # 刻度
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    # 隐藏坐标轴数字，只保留格子
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # 打开网格
    ax.grid(True)
    # 方格等长
    ax.set_aspect('equal')
    # 图形表头
    ax.set_title(f"2D {env.width}*{env.height} Matrix rollout")
    # 设置物体/智能体显示
    x, y = positions[0]
    agent_frames = load_gif_frames(agent_img_path)
    imagebox = OffsetImage(agent_frames[0], zoom=0.05)
    agent_avatar = AnnotationBbox(
        imagebox,
        (x + 0.5, y + 0.5),
        frameon=False
    )
    ax.add_artist(agent_avatar)
    # 设置终点位置
    max_x, max_y = env.max_pos
    destination_frames = load_gif_frames(destination_img_path)
    des_imagebox = OffsetImage(destination_frames[0], zoom=0.05)
    des_avatar = AnnotationBbox(
        des_imagebox,
        (max_x + 0.5, max_y + 0.5),
        frameon=False
    )
    ax.add_artist(des_avatar)
    # 设置胜利结束动画显示
    ending_frames = load_gif_frames(ending_img_path)
    end_imagebox = OffsetImage(ending_frames[0], zoom=0.4)
    end_avatar = AnnotationBbox(
        end_imagebox,
        (max_x/2, max_y/2),
        frameon=False
    )    
    ax.add_artist(end_avatar)
    end_avatar.set_visible(False)
    # 设置物体动作
    arrow = ACTION_ARROW.get(actions[0], "?")
    agent_action, = ax.plot(x + 0.7, y + 0.7, marker=arrow, markersize=10)
    # 计步器
    step_test = ax.text(
        0.02, 0.98, "Step: 0",
        transform=ax.transAxes,
        va="top", ha="left"
    )

    moving_time = len(positions)

    def update(frame_idx): 
        if frame_idx < moving_time:
            # 更新物体位置显示
            x, y = positions[frame_idx]
            gif_idx = frame_idx % len(agent_frames)
            agent_avatar.xybox = (x + 0.5, y +0.5)
            agent_avatar.offsetbox.set_data(agent_frames[gif_idx])
            # 更新动作显示
            if frame_idx == 0:
                a = 9
            else:
                a = actions[frame_idx - 1]
            arrow = ACTION_ARROW.get(a, "?")
            agent_action.set_data([x + 0.7], [y + 0.7])
            agent_action.set_marker(arrow)
            # 更新步数显示
            step_test.set_text(f"Step: {frame_idx}")
            # 终点gif跳变更新
            des_gif_idx = frame_idx % len(destination_frames)
            des_avatar.offsetbox.set_data(destination_frames[des_gif_idx])
        elif frame_idx >= moving_time and terminated == True:
            agent_avatar.set_visible(False)
            des_avatar.set_visible(False)
            end_avatar.set_visible(True)
            end_gif_idx = frame_idx % len(ending_frames)
            end_avatar.offsetbox.set_data(ending_frames[end_gif_idx])

        return agent_avatar, agent_action, step_test, des_avatar

    ani = FuncAnimation(
        fig,
        update,
        frames=moving_time + ending_time,
        interval = 500,
        blit=False,
        repeat=False
    )

    # plt.show()
    ani.save(gif_path, writer="pillow")
    return ani

def load_gif_frames(gif_path):
    img = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = img.convert("RGBA")
            frames.append(np.array(frame))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames