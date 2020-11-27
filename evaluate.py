from environment import CartPoleEnv
from dataset import raw_states_to_torch
from model import Net

import torch
import numpy as np
import os
import time


def evaluate_in_environment(net, mean, std, nr_iters=1, render=False):
    eval_env = CartPoleEnv()
    with torch.no_grad():
        success = np.zeros(nr_iters)
        for it in range(nr_iters):
            is_fine = False
            episode_length_counter = 0
            new_state = eval_env.state

            # To make the randomization stronger, so the performance is better
            # visible:
            if render:
                eval_env.state = mean + (
                    np.random.rand(len(mean)) - .5
                ) * 2 * std

            # Start balancing
            while not is_fine:
                # Transform state in the same way as the training data
                # and normalize
                torch_state = raw_states_to_torch(
                    new_state, mean=mean, std=std
                )
                # Predict optimal action:
                action = torch.sigmoid(net(torch_state))
                action = (action.item() - .5) * 3

                # run action in environment
                new_state, _, is_fine, _ = eval_env._step(action)
                if render:
                    eval_env._render()
                    time.sleep(.2)
                # track number of timesteps until failure
                episode_length_counter += 1
                if episode_length_counter > 250:
                    break
            success[it] = episode_length_counter
            eval_env._reset()
    return success


if __name__ == "__main__":
    # make as args:
    MODEL_NAME = "without_normalize"  # "best_model_2"

    net = torch.load(os.path.join("models", MODEL_NAME, "model_pendulum"))
    net.eval()

    data_arr = np.load(os.path.join("models", MODEL_NAME, "state_data.npy"))
    mean = np.mean(data_arr, axis=0)
    std = np.std(data_arr, axis=0)

    success = evaluate_in_environment(net, mean, std, render=True)
