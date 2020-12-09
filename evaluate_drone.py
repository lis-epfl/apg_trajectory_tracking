import os
import time
import argparse
import json
import numpy as np
import torch

from environments.drone_env import QuadRotorEnvBase
from dataset import raw_states_to_torch
from models.resnet_like_model import Net


class QuadEvaluator():

    def __init__(self, model, std):
        self.std = std
        self.net = model

    def stabilize(self, nr_iters=1, render=False, max_time=100):
        with torch.no_grad():
            eval_env = QuadRotorEnvBase()
            collect_runs = list()
            for _ in range(nr_iters):
                time_stable = 0
                eval_env.reset()
                current_np_state = eval_env._state.as_np
                stable = True
                while stable and time_stable < max_time:
                    current_torch_state = torch.from_numpy(
                        np.array([current_np_state])
                    ).float()
                    suggested_action = self.net(current_torch_state)
                    suggested_action = torch.sigmoid(suggested_action
                                                     )[0].numpy()
                    current_np_state, stable = eval_env.step(suggested_action)
                    if render:
                        eval_env.render()
                        time.sleep(.1)
                    time_stable += 1
                collect_runs.append(time_stable)
        return np.mean(collect_runs), np.std(collect_runs)


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="test_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join("trained_models", "drone", model_name)

    # load std or other parameters from json
    with open(os.path.join(model_path, "std.json"), "r") as outfile:
        std = np.array(json.load(outfile)["std"])

    net = torch.load(os.path.join(model_path, "model_quad"))
    net.eval()

    evaluator = QuadEvaluator(net, std)
    success_mean, success_std = evaluator.stabilize(nr_iters=3, render=True)
    print(success_mean, success_std)
