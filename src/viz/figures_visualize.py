import matplotlib.pyplot as plt

def draw_learning_curve(episode_step) -> None:

    plt.plot(episode_step)
    plt.xlabel("Episodes")
    plt.ylabel("step to reach Goal")
    plt.title("Learning Curve")
    plt.show()


