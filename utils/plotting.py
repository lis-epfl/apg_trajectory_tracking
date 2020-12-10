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
