from environment import CartPoleEnv
from dataset import raw_states_to_torch
from model import Net

import torch
import numpy as np
import os
import time
import argparse
from control_loss import control_loss_function


class Evaluator:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def make_swingup(self, net, nr_iters=3, max_iters=300, render=False):
        """
        Check if the pendulum can make a swing up
        """
        eval_env = CartPoleEnv()
        with torch.no_grad():
            success = np.zeros(nr_iters)
            for it in range(nr_iters):
                # np.random.seed(it + 300)
                random_hanging_state = (np.random.rand(4) - .5)
                random_hanging_state[2] = (-1) * (
                    (np.random.rand() > .5) * 2 - 1
                ) * (1 - (np.random.rand() * .2)) * np.pi
                eval_env.state = random_hanging_state
                new_state = eval_env.state

                # Start balancing
                made_it = False
                for _ in range(max_iters):
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(
                        new_state, mean=self.mean, std=self.std
                    )
                    # Predict optimal action:
                    predicted_action = net(torch_state)
                    action_seq = torch.sigmoid(predicted_action) - .5
                    # print([round(act, 2) for act in action_seq[0].numpy()])
                    for action in action_seq[0].numpy():
                        # run action in environment
                        new_state, _, _, _ = eval_env._step(action)
                        # print(new_state)
                        if render:
                            eval_env._render()
                            time.sleep(.1)
                        # check only whether it was able to swing up the pendulum
                        if np.abs(new_state[2]) < np.pi / 15 and not render:
                            made_it = 1
                            break
                success[it] = made_it
                eval_env._reset()
        return success

    def run_for_fixed_length(
        self, net, episode_length=100, nr_iters=1, render=False
    ):
        """
        Measure average angle deviation
        """
        # return only the angles
        eval_env = CartPoleEnv()
        with torch.no_grad():
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for it in range(nr_iters):
                new_state = eval_env.state
                # To make the randomization stronger, so the performance is better
                # visible:
                if render:
                    eval_env.state = self.mean + (
                        np.random.rand(len(self.mean)) - .5
                    ) * 2 * self.std

                angles = list()
                # Start balancing
                for _ in range(episode_length):
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(
                        new_state, mean=self.mean, std=self.std
                    )
                    # Predict optimal action:
                    action = torch.sigmoid(net(torch_state))[0, 0]
                    action = action.item() - .5

                    # run action in environment
                    new_state, _, is_fine, _ = eval_env._step(action)
                    angles.append(np.absolute(new_state[2]))
                    if render:
                        eval_env._render()
                        time.sleep(.2)
                avg_angle[it] = np.mean(angles)
                eval_env._reset()
        return avg_angle

    def evaluate_in_environment(self, net, nr_iters=1, render=False):
        """
        Measure success --> how long can we balance the pole on top
        """
        eval_env = CartPoleEnv()
        with torch.no_grad():
            success = np.zeros(nr_iters)
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for it in range(nr_iters):
                eval_env.state = (np.random.rand(4) - .5) * .1
                is_fine = False
                episode_length_counter = 0
                new_state = eval_env.state

                # To make the randomization stronger, so the performance is better
                # visible:
                if render:
                    eval_env.state = self.mean + (
                        np.random.rand(len(self.mean)) - .5
                    ) * .6 * self.std

                angles = list()
                # Start balancing
                while not is_fine:
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(
                        new_state, mean=self.mean, std=self.std
                    )
                    # Predict optimal action:
                    action = torch.sigmoid(net(torch_state))[0, 0]
                    action = action.item() - .5

                    # run action in environment
                    new_state, _, is_fine, _ = eval_env._step(action)
                    angles.append(np.absolute(new_state[2]))
                    if render:
                        eval_env._render()
                        time.sleep(.1)
                    # track number of timesteps until failure
                    episode_length_counter += 1
                    if episode_length_counter > 250:
                        break
                avg_angle[it] = np.mean(angles)
                success[it] = episode_length_counter
                eval_env._reset()
        return success, avg_angle


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="theta_max_normalize",
        help="Directory of model"
    )
    args = parser.parse_args()

    MODEL_NAME = args.model  # "theta_max_normalize"  # "best_model_2"

    net = torch.load(os.path.join("models", MODEL_NAME, "model_pendulum"))
    net.eval()

    data_arr = np.load(os.path.join("models", MODEL_NAME, "state_data.npy"))
    mean = np.mean(data_arr, axis=0)
    std = np.std(data_arr, axis=0)
    # mean = np.zeros(4)
    # std = np.ones(4)

    evaluator = Evaluator(mean, std)
    # angles = evaluator.run_for_fixed_length(net, render=True)
    success = evaluator.make_swingup(net, render=True)
    # success, angles = evaluator.evaluate_in_environment(net, render=True)
