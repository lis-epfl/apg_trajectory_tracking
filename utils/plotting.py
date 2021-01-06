import os
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(loss, save_path):
    """
    Simple plot of training loss
    """
    plt.figure(figsize=(15, 8))
    plt.plot(loss)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.savefig(os.path.join(save_path, "loss.png"))


def plot_success(means, stds, save_path):
    episode_length_mean = np.array(means)
    episode_length_std = np.array(stds)
    plt.figure(figsize=(20, 10))
    x = np.arange(len(episode_length_mean))
    plt.plot(x, episode_length_mean, '-')
    plt.fill_between(
        x,
        episode_length_mean - episode_length_std,
        episode_length_mean + episode_length_std,
        alpha=0.2
    )
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Average episode length", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(os.path.join(save_path, "performance.png"))


def plot_state_variables(collect_data, save_path=None):
    """
    Plot a collection of state by plotting position and velocities over time
    """
    collect_data = np.delete(np.array(collect_data), [9, 10, 11, 12], axis=1)
    collect_data[:, 2] = collect_data[:, 2] - 2
    print(collect_data.shape)
    labels = [
        "roll", "pitch", "yaw", "x", "y", "z", "vel_x", "vel_y", "vel_z",
        "vel_roll", "vel_pitch", "vel_yaw"
    ]
    plt.figure(figsize=(20, 10))
    for i in range(collect_data.shape[1]):
        plt.plot(collect_data[:, i], label=labels[i])
    plt.legend(fontsize="15")
    plt.ylim(-2, 2)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_position(collect_data, save_path=None):
    """
    Plot only how the position evolves
    """
    plt.figure(figsize=(20, 10))
    labels = ["x", "y", "z"]
    for i in range(3):
        plt.plot(collect_data[:, i], label=labels[i])
    plt.legend(fontsize="15")
    plt.ylim(-1, 1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
