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
    collect_data = np.array(collect_data)
    plt.figure(figsize=(20, 10))
    labels = ["x", "y", "z"]
    for i in range(3):
        plt.plot(collect_data[:, i], label=labels[i])
    plt.legend(fontsize="15")
    # plt.ylim(-1, 1)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_trajectory(knots, states, save_path):
    states = np.array(states)
    buffer = 0.2
    plt.figure(figsize=(20, 10))
    min_z = np.min(knots[:, 2] - .5)
    max_z = np.max(knots[:, 2] + .5)
    normed_z = (states[:, 2] - min_z) / (max_z - min_z)
    # scatter states
    plt.scatter(states[:, 0], states[:, 1], s=50 * normed_z, c="green")
    # scatter trajectory
    normed_knot_z = (knots[:, 2] - min_z) / (max_z - min_z)
    plt.scatter(knots[:, 0], knots[:, 1], s=50 * normed_knot_z, c="red")
    plt.scatter(
        knots[-1, 0],
        knots[-1, 1],
        s=50 * normed_knot_z[-1],
        c="blue",
        label="target"
    )

    plt.xlim(np.min(knots[:, 0]) - buffer, np.max(knots[:, 0]) + buffer)
    plt.ylim(np.min(knots[:, 1]) - buffer, np.max(knots[:, 1]) + buffer)
    # plt.xlim(-1,1)
    plt.savefig(save_path)


def plot_loss_episode_len(
    episode_length_mean, episode_length_std, loss_list, save_path=None
):
    """
    Plot episode length and losses together in one plot
    """
    x = np.arange(len(episode_length_mean))
    fig, ax1 = plt.subplots(figsize=(20, 10))

    color = 'tab:red'
    ax1.set_xlabel("Epoch", fontsize=18)
    ax1.plot(x, episode_length_mean, '-', color=color, label="Performance")
    ax1.fill_between(
        x,
        episode_length_mean - episode_length_std,
        episode_length_mean + episode_length_std,
        alpha=0.2,
        color=color
    )
    ax1.set_ylabel("Average episode length", color=color, fontsize=18)
    # ax1.tick_params(axis='x', fontsize=18)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color, fontsize=18)
    ax2.plot(loss_list, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
