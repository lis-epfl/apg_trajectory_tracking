import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.wing_env import SimpleWingEnv, run_wing_flight
from neural_control.utils.plotting import plot_wing_pos_3d
from neural_control.dataset import WingDataset
from evaluate_drone import load_model_params
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.controllers.mpc import MPC

ACTION_DIM = 2


class FixedWingEvaluator:
    """
    Evaluate performance of the fixed wing drone
    """

    def __init__(self, controller, dt=0.01, horizon=1, render=0, **kwargs):
        self.controller = controller
        self.dt = dt
        self.horizon = horizon
        self.render = render
        self.eval_env = SimpleWingEnv(dt)

    def fly_to_point(self, target_points, max_steps=1000):
        self.eval_env.zero_reset()
        if self.render:
            self.eval_env.drone_render_object.set_target(target_points)

        # first target
        current_target_ind = 0

        state = self.eval_env._state
        stable = True
        drone_traj = []
        while stable and len(drone_traj) < max_steps:
            current_target = target_points[current_target_ind]
            action = self.controller.predict_actions(state, current_target)
            # print(action[0])
            state, stable = self.eval_env.step(action[0])
            # print(state[2])
            if self.render:
                self.eval_env.render()
                time.sleep(.05)
            drone_traj.append(np.concatenate((state, action[0])))
            if state[0] > current_target[0]:
                if current_target_ind < len(target_points) - 1:
                    current_target_ind += 1
                else:
                    break
        return np.array(drone_traj)

    def run_eval(self, nr_test, return_dists=False):
        min_dists = []
        for i in range(nr_test):
            # target_point = [
            #     np.random.rand(3) * np.array([60, 10, 10]) +
            #     np.array([30, -5, -5])
            # ]
            traj_test = run_wing_flight(
                self.eval_env, traj_len=300, dt=self.dt, render=0
            )
            target_point = [traj_test[-1, :3]]
            drone_traj = self.fly_to_point(target_point)
            last_x_points = drone_traj[-20:, :3]
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


def load_model(model_path, epoch="", horizon=10, dt=0.05, **kwargs):
    """
    Load model and corresponding parameters
    """
    if "mpc" not in model_path:
        # load std or other parameters from json
        net, param_dict = load_model_params(
            model_path, "model_wing", epoch=epoch
        )
        dataset = WingDataset(100, **param_dict)

        controller = FixedWingNetWrapper(net, dataset, **param_dict)
    else:
        controller = MPC(horizon, dt, dynamics="fixed_wing")
    return controller


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="current_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    args = parser.parse_args()

    # parameters
    params = {"render": 1, "dt": 0.05, "horizon": 10}

    # load model
    model_name = args.model
    model_path = os.path.join("trained_models", "wing", model_name)

    controller = load_model(
        model_path, epoch=args.epoch, name="model_wing", **params
    )

    evaluator = FixedWingEvaluator(controller, **params)

    # only run evaluation without render
    # tic = time.time()
    # out_path = "../presentations/intel_meeting_10_03"
    # evaluator.render = 0
    # dists_from_target = evaluator.run_eval(nr_test=100, return_dists=True)
    # np.save(os.path.join(out_path, "dists_mpc_last.npy"), dists_from_target)
    # print("time for 100 trajectories", time.time() - tic)
    # exit()

    target_point = [[70, 10, 0]]  # , [140, -4, -3]]

    # RUN
    drone_traj = evaluator.fly_to_point(target_point, max_steps=1000)
    print(drone_traj[-1])
    print(drone_traj.shape)
    # np.save(os.path.join(model_path, "drone_traj.npy"), drone_traj)

    evaluator.eval_env.close()

    # EVAL
    plot_wing_pos_3d(
        drone_traj,
        target_point,
        save_path=os.path.join(model_path, "coords.png")
    )
