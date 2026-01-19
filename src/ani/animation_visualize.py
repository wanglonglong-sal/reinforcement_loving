import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image


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
    # Initialize canvas
    fig, ax = plt.subplots()

    ax.set_xlim(-0.5, max_pos + 0.5)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks(range(max_pos + 1))
    ax.set_title("LineWorld rollout")
    # Set goal position
    ax.scatter([max_pos], [0], marker="*", s=250)
    # Set object/agent
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
    # Initialize canvas
    fig, ax = plt.subplots()

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    # Ticks
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    # Hide axis labels, keep only grid cells
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Enable grid
    ax.grid(True)
    # Equal aspect ratio for grid cells
    ax.set_aspect('equal')
    # Figure title
    ax.set_title(f"2D {env.width}*{env.height} Matrix rollout")
    # Set goal position
    max_x, max_y = env.max_pos
    ax.scatter(max_x + 0.5, max_y + 0.5, marker="*", s=250)
    # Set object/agent display
    x, y = positions[0]
    agent_dot, = ax.plot(x + 0.5, y + 0.5, marker="o", markersize=14)
    # Set action display
    arrow = ACTION_ARROW.get(actions[1], "?")
    agent_action, = ax.plot(x + 0.5, y + 0.5, marker=arrow, markersize=10)
    # Step counter
    step_test = ax.text(
        0.02, 0.98, "Step: 0",
        transform=ax.transAxes,
        va="top", ha="left"
    )

    def update(frame_idx):
        # Update agent position display
        x, y = positions[frame_idx]
        agent_dot.set_data([x + 0.5], [y + 0.5])
        # Update action display
        if frame_idx == 0:
            a = 9
        else:
            a = actions[frame_idx - 1]
        arrow = ACTION_ARROW.get(a, "?")
        agent_action.set_data([x + 0.5], [y + 0.5])
        agent_action.set_marker(arrow)
        # Update step count display
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
    # Initialize canvas
    fig, ax = plt.subplots()
    # Width, height
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    # Ticks
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    # Hide axis labels, keep only grid cells
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Enable grid
    ax.grid(True)
    # Equal aspect ratio for grid cells
    ax.set_aspect('equal')
    # Figure title
    ax.set_title(f"2D {env.width}*{env.height} Matrix rollout")
    # Set object/agent display
    x, y = positions[0]
    agent_frames = load_gif_frames(agent_img_path)
    imagebox = OffsetImage(agent_frames[0], zoom=0.05)
    agent_avatar = AnnotationBbox(
        imagebox,
        (x + 0.5, y + 0.5),
        frameon=False
    )
    ax.add_artist(agent_avatar)
    # Set goal position
    max_x, max_y = env.max_pos
    destination_frames = load_gif_frames(destination_img_path)
    des_imagebox = OffsetImage(destination_frames[0], zoom=0.05)
    des_avatar = AnnotationBbox(
        des_imagebox,
        (max_x + 0.5, max_y + 0.5),
        frameon=False
    )
    ax.add_artist(des_avatar)
    # Set victory ending animation display
    ending_frames = load_gif_frames(ending_img_path)
    end_imagebox = OffsetImage(ending_frames[0], zoom=0.4)
    end_avatar = AnnotationBbox(
        end_imagebox,
        (max_x/2, max_y/2),
        frameon=False
    )    
    ax.add_artist(end_avatar)
    end_avatar.set_visible(False)
    # Set action display
    arrow = ACTION_ARROW.get(actions[0], "?")
    agent_action, = ax.plot(x + 0.7, y + 0.7, marker=arrow, markersize=10)
    # Step counter
    step_test = ax.text(
        0.02, 0.98, "Step: 0",
        transform=ax.transAxes,
        va="top", ha="left"
    )

    moving_time = len(positions)

    def update(frame_idx): 
        if frame_idx < moving_time:
            # Update agent position display
            x, y = positions[frame_idx]
            gif_idx = frame_idx % len(agent_frames)
            agent_avatar.xybox = (x + 0.5, y +0.5)
            agent_avatar.offsetbox.set_data(agent_frames[gif_idx])
            # Update action display
            if frame_idx == 0:
                a = 9
            else:
                a = actions[frame_idx - 1]
            arrow = ACTION_ARROW.get(a, "?")
            agent_action.set_data([x + 0.7], [y + 0.7])
            agent_action.set_marker(arrow)
            # Update step count display
            step_test.set_text(f"Step: {frame_idx}")
            # Update goal gif frame
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

def animate_position_2d_img_ran(env, positions, actions, gif_path, agent_img_path, destination_img_path, ending_img_path, terminated, ending_time):
    # Initialize canvas
    fig, ax = plt.subplots()
    # Width, height
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    # Ticks
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    # Hide axis labels, keep only grid cells
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # Enable grid
    ax.grid(True)
    # Equal aspect ratio for grid cells
    ax.set_aspect('equal')
    # Figure title
    ax.set_title(f"2D {env.width}*{env.height} Matrix rollout")
    # Set object/agent display
    x, y = positions[0]
    agent_frames = load_gif_frames(agent_img_path)
    imagebox = OffsetImage(agent_frames[0], zoom=0.05)
    agent_avatar = AnnotationBbox(
        imagebox,
        (x + 0.5, y + 0.5),
        frameon=False
    )
    ax.add_artist(agent_avatar)
    # Set goal position
    goal_x, goal_y = env.goal_pos
    destination_frames = load_gif_frames(destination_img_path)
    des_imagebox = OffsetImage(destination_frames[0], zoom=0.05)
    des_avatar = AnnotationBbox(
        des_imagebox,
        (goal_x + 0.5, goal_y + 0.5),
        frameon=False
    )
    ax.add_artist(des_avatar)
    # Set victory ending animation display
    ending_frames = load_gif_frames(ending_img_path)
    end_imagebox = OffsetImage(ending_frames[0], zoom=0.4)
    end_avatar = AnnotationBbox(
        end_imagebox,
        (goal_x/2, goal_y/2),
        frameon=False
    )    
    ax.add_artist(end_avatar)
    end_avatar.set_visible(False)
    # Set action display
    arrow = ACTION_ARROW.get(actions[0], "?")
    agent_action, = ax.plot(x + 0.7, y + 0.7, marker=arrow, markersize=10)
    # Step counter
    step_test = ax.text(
        0.02, 0.98, "Step: 0",
        transform=ax.transAxes,
        va="top", ha="left"
    )

    moving_time = len(positions)

    def update(frame_idx): 
        if frame_idx < moving_time:
            # Update agent position display
            x, y = positions[frame_idx]
            gif_idx = frame_idx % len(agent_frames)
            agent_avatar.xybox = (x + 0.5, y +0.5)
            agent_avatar.offsetbox.set_data(agent_frames[gif_idx])
            # Update action display
            if frame_idx == 0:
                a = 9
            else:
                a = actions[frame_idx - 1]
            arrow = ACTION_ARROW.get(a, "?")
            agent_action.set_data([x + 0.7], [y + 0.7])
            agent_action.set_marker(arrow)
            # Update step count display
            step_test.set_text(f"Step: {frame_idx}")
            # Update goal gif frame
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
