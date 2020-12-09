import os
import matplotlib.pyplot as plt


def plot_loss(loss, save_path):
    """
    Simple plot of training loss
    """
    plt.figure(figsize=(15, 8))
    plt.plot(loss)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.savefig(os.path.join(save_path, "loss.png"))
