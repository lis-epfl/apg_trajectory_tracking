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
    labels = [
        "x", "y", "z", "roll", "pitch", "yaw", "vel_x", "vel_y", "vel_z",
        "vel_roll", "vel_pitch", "vel_yaw"
    ]
    plt.figure(figsize=(20, 10))
    for i in range(collect_data.shape[1]):
        plt.plot(collect_data[:, i], label=labels[i])
    plt.legend(fontsize="15")
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


def print_state_ref_div(np_state, np_ref):
    """
    pretty printing of target trajectory and actual trajectory
    """
    np.set_printoptions(suppress=True, precision=3)
    traj_len = len(np_ref)
    print("Positions:")
    for i in range(traj_len):
        print(np_state[i, :3], "ref:", np_ref[i, :3])

    print("Attitudes:")
    for i in range(traj_len):
        print(np_state[i, 3:6], "ref:", np_ref[i, 3:6])

    print("Velocities:")
    for i in range(traj_len):
        print(np_state[i, 6:9], "ref:", np_ref[i, 6:9])

    print("Body rates:")
    for i in range(traj_len):
        print(np_state[i, 9:], "ref:", np_ref[i, 9:])


def plot_wing_pos(states, targets, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(states[:, 0], states[:, 1], label="x-h position")
    for target in targets:
        plt.scatter(target[0], target[1], label="target", c="red")
    plt.legend(fontsize=20)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_drone_ref_coords(
    drone_trajectory, reference_trajectory, save_path=None
):
    if len(drone_trajectory) == 0:
        return 0
    quad_pos = np.hstack([drone_trajectory, reference_trajectory])
    plt.figure(figsize=(15, 8))
    labels = ["quad_x", "quad_y", "quad_z", "ref_x", "ref_y", "ref_z"]
    cols = ["-r", "-b", "-g", "--r", "--b", "--g"]
    quad_pos = np.swapaxes(np.array(quad_pos), 1, 0)
    for i, data in enumerate(quad_pos):
        plt.plot(data, cols[i], label=labels[i])
    plt.legend(fontsize=15)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def plot_trajectory(knots, states, save_path, fixed_axis=2):
    if len(states) == 0:
        return 0
    leftover = [0, 1, 2]
    del leftover[fixed_axis]
    states = np.array(states)
    buffer = 0.5
    plt.figure(figsize=(10, 10))
    min_z = np.min(knots[:, fixed_axis] - .5)
    max_z = np.max(knots[:, fixed_axis] + .5)
    normed_z = (states[:, fixed_axis] - min_z) / (max_z - min_z)
    # scatter states
    plt.scatter(
        states[:, leftover[0]],
        states[:, leftover[1]],
        s=50 * normed_z,
        c="green",
        label="drone trajectory"
    )
    # scatter trajectory
    normed_knot_z = (knots[:, fixed_axis] - min_z) / (max_z - min_z)
    plt.scatter(
        knots[:, leftover[0]],
        knots[:, leftover[1]],
        s=50 * normed_knot_z,
        c="red",
        label="reference"
    )
    plt.scatter(
        knots[-1, leftover[0]],
        knots[-1, leftover[1]],
        s=50 * normed_knot_z[-1],
        c="blue",
        label="target"
    )

    plt.xlim(
        np.min(knots[:, leftover[0]]) - buffer,
        np.max(knots[:, leftover[0]]) + buffer
    )
    plt.ylim(
        np.min(knots[:, leftover[1]]) - buffer,
        np.max(knots[:, leftover[1]]) + buffer
    )
    # plt.xlim(-1,1)
    plt.legend()
    plt.savefig(save_path)


def plot_loss_episode_len(
    episode_length_mean, episode_length_std, loss_list, save_path=None
):
    """
    Plot episode length and losses together in one plot
    """
    episode_length_mean = np.array(episode_length_mean)
    episode_length_std = np.array(episode_length_std)
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


def plot_suc_by_dist(distances, success_mean_list, save_path):
    """
    Plot success rate by the distance of the drone from the target
    """
    plt.plot(distances, success_mean_list)
    plt.xlabel("distance of drone ")
    plt.ylabel("Average episode length")
    plt.ylim(0, 200)
    plt.xlim(0, 0.8)
    plt.savefig(os.path.join(save_path, "succ_by_dist.png"))
