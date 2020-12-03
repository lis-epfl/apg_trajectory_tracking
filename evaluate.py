from environment import CartPoleEnv
from dataset import raw_states_to_torch
from model import Net

import torch
import numpy as np
import os
import time
import argparse
from control_loss import control_loss_function

data_collection = []


class Evaluator:

    def __init__(self, std):
        self.std = std

    def make_swingup(
        self, net, nr_iters=3, max_iters=40, success_over=5, render=False
    ):
        """
        Check if the pendulum can make a swing up
        """
        # average over 50 runs # 10 is length of action sequence
        success = []  # np.zeros((success_over * 10 * nr_iters, 4))
        eval_env = CartPoleEnv()
        with torch.no_grad():
            for it in range(nr_iters):
                # np.random.seed(it + 300)
                random_hanging_state = (np.random.rand(4) - .5)
                random_hanging_state[2] = (-1) * (
                    (np.random.rand() > .5) * 2 - 1
                ) * (1 - (np.random.rand() * .2)) * np.pi
                random_hanging_state[0] = 0
                eval_env.state = random_hanging_state
                new_state = eval_env.state

                # Start balancing
                for j in range(max_iters + success_over):
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(new_state, std=self.std)
                    # Predict optimal action:
                    predicted_action = net(torch_state)
                    # control_loss_function(
                    #     predicted_action, torch_state, printout=1
                    # )
                    action_seq = torch.sigmoid(predicted_action) - .5
                    # print([round(act, 2) for act in action_seq[0].numpy()])
                    # print("state", new_state)
                    # print("new action seq", action_seq[0].numpy())
                    # print()
                    for action in action_seq[0].numpy():
                        # run action in environment
                        new_state, _, _, _ = eval_env._step(action)
                        data_collection.append(new_state)
                        # print(new_state)
                        if render:
                            eval_env._render()
                            time.sleep(.1)
                        if j >= max_iters:
                            success.append(new_state)
                        # check only whether it was able to swing up the pendulum
                        # if np.abs(new_state[2]) < np.pi / 15 and not render:
                        #     made_it = 1
                        #     break
                # success[it] = made_it
                eval_env._reset()
        success = np.array(success)
        mean_rounded = [round(m, 2) for m in np.mean(success, axis=0)]
        std_rounded = [round(m, 2) for m in np.std(success, axis=0)]
        return mean_rounded, std_rounded

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
                    eval_env.state = (
                        np.random.rand(len(self.std)) - .5
                    ) * 2 * self.std

                angles = list()
                # Start balancing
                for _ in range(episode_length):
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(new_state, std=self.std)
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
                    eval_env.state = (
                        np.random.rand(len(self.std)) - .5
                    ) * .4 * self.std

                angles = list()
                # Start balancing
                while not is_fine:
                    # Transform state in the same way as the training data
                    # and normalize
                    torch_state = raw_states_to_torch(new_state, std=self.std)
                    # Predict optimal action:
                    action_seq = torch.sigmoid(net(torch_state)) - .5
                    for action in action_seq[0].numpy():
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
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    MODEL_NAME = args.model  # "theta_max_normalize"  # "best_model_2"

    net = torch.load(os.path.join("models", MODEL_NAME, "model_pendulum"))
    net.eval()

    data_arr = np.load(os.path.join("models", MODEL_NAME, "state_data.npy"))
    std = np.std(data_arr, axis=0)

    evaluator = Evaluator(std)
    # angles = evaluator.run_for_fixed_length(net, render=True)
    # success, angles = evaluator.evaluate_in_environment(net, render=True)
    try:
        _ = evaluator.make_swingup(net, max_iters=100, render=True)
    except KeyboardInterrupt:
        data_collection = np.array(data_collection)
        do_it = input(
            f"Name to save collection of data with size {data_collection.shape}?"
        )
        if len(do_it) > 2:
            np.save(os.path.join("data", do_it + ".npy"), data_collection)
