import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.wing_env import SimpleWingEnv, run_wing_flight
from neural_control.utils.plotting import plot_wing_pos
from neural_control.dataset import WingDataset
from evaluate_drone import load_model

ACTION_DIM = 2


class FixedWingEvaluator:
    """
    Evaluate performance of the fixed wing drone
    """

    def __init__(self, model, dataset, dt=0.01, horizon=1, render=0, **kwargs):
        self.net = model
        self.dt = dt
        self.dataset = dataset
        self.horizon = horizon
        self.eval_env = SimpleWingEnv(dt)

    def predict_actions(self, state, ref_state):
        normed_state, _, normed_ref, _ = self.dataset.prepare_data(
            state, ref_state
        )
        with torch.no_grad():
            suggested_action = self.net(normed_state, normed_ref)
            suggested_action = torch.sigmoid(suggested_action)[0]

            suggested_action = torch.reshape(
                suggested_action, (self.horizon, ACTION_DIM)
            )
        return suggested_action[0].detach().numpy()

    def fly_to_point(self, target_point):
        self.eval_env.zero_reset()

        state = self.eval_env._state
        stable = True
        drone_traj = []
        while stable and state[0] < target_point[0] + 1:
            action = self.predict_actions(state, target_point)
            # np.random.rand(2)
            # self.predict_actions(state, target_point)
            state, stable = self.eval_env.step(tuple(action))
            drone_traj.append(state)

        return np.array(drone_traj)

    def run_eval(self, nr_test):
        min_dists = []
        for i in range(nr_test):
            test_traj = run_wing_flight(num_traj=1, traj_len=350, dt=self.dt)
            where_to_sample = int(250 + np.random.rand(1) * 100)
            target_point = test_traj[0, where_to_sample, :2]
            drone_traj = self.fly_to_point(target_point)
            last_x_points = drone_traj[-5:, :2]
            last_x_dists = [
                np.linalg.norm(target_point - p) for p in last_x_points
            ]
            min_dists.append(np.min(last_x_dists))
        mean_err = np.mean(min_dists)
        std_err = np.std(min_dists)
        print(f"Average error: {mean_err} (std: {std_err})")
        return mean_err, std_err


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
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    args = parser.parse_args()

    # rendering
    render = 1

    # load model
    model_name = args.model
    model_path = os.path.join("trained_models", "wing", model_name)
    net, param_dict = load_model(
        model_path, epoch=args.epoch, name="model_wing"
    )

    dataset = WingDataset(100, **param_dict)
    evaluator = FixedWingEvaluator(
        net, dataset=dataset, render=render, **param_dict
    )

    test_traj = run_wing_flight(num_traj=1, traj_len=350, dt=param_dict["dt"])
    target_point = test_traj[0, 300, :2]
    print("target_point", target_point)

    # RUN
    drone_traj = evaluator.fly_to_point(target_point)

    # EVAL
    plot_wing_pos(
        drone_traj,
        target=target_point,
        save_path=os.path.join(model_path, "coords.png")
    )
