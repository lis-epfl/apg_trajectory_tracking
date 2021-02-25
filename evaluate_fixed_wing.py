import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.wing_env import SimpleWingEnv, run_wing_flight
from neural_control.utils.plotting import plot_wing_pos
from neural_control.dataset import WingDataset
from evaluate_drone import load_model_params

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
        self.render = render
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
        if self.render:
            self.eval_env.drone_render_object.set_target(target_point)

        state = self.eval_env._state
        stable = True
        drone_traj = []
        while stable and state[0] < target_point[0] + .5:
            action = self.predict_actions(state, target_point)
            # np.random.rand(2)
            # self.predict_actions(state, target_point)
            state, stable = self.eval_env.step(tuple(action))
            if self.render:
                self.eval_env.render()
            drone_traj.append(state)

        return np.array(drone_traj)

    def run_eval(self, nr_test, return_dists=False):
        min_dists = []
        for i in range(nr_test):
            test_traj = run_wing_flight(num_traj=1, traj_len=350, dt=self.dt)
            where_to_sample = int(250 + np.random.rand(1) * 100)
            target_point = test_traj[0, where_to_sample, :2]
            drone_traj = self.fly_to_point(target_point)
            last_x_points = drone_traj[-20:, :2]
            last_x_dists = [
                np.linalg.norm(target_point - p) for p in last_x_points
            ]
            min_dists.append(np.min(last_x_dists))
        mean_err = np.mean(min_dists)
        std_err = np.std(min_dists)
        print("Average error: %3.2f (%3.2f)" % (mean_err, std_err))
        if return_dists:
            return np.array(min_dists)
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
    net, param_dict = load_model_params(
        model_path, epoch=args.epoch, name="model_wing"
    )

    dataset = WingDataset(100, **param_dict)
    evaluator = FixedWingEvaluator(
        net, dataset=dataset, render=render, **param_dict
    )

    # only run evaluation without render
    # out_path = "../presentations/intel_meeting_26_02"
    # evaluator.render = 0
    # dists_from_target = evaluator.run_eval(nr_test=100, return_dists=True)
    # np.save(os.path.join(out_path, "dists_target.npy"), dists_from_target)
    # exit()

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
