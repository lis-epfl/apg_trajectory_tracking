import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.wing_env import SimpleWingEnv, run_wing_flight
from neural_control.plotting import plot_wing_pos_3d
from neural_control.dataset import WingDataset
from evaluate_drone import load_model_params
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.controllers.mpc import MPC
from neural_control.trajectory.q_funcs import project_to_line


class FixedWingEvaluator:
    """
    Evaluate performance of the fixed wing drone
    """

    def __init__(
        self,
        controller,
        dt=0.01,
        horizon=1,
        render=0,
        thresh_div=10,
        thresh_stable=0.8,
        **kwargs
    ):
        self.controller = controller
        self.dt = dt
        self.horizon = horizon
        self.render = render
        self.thresh_div = thresh_div
        self.thresh_stable = thresh_stable
        self.eval_env = SimpleWingEnv(dt)
        self.des_speed = 11.5

    def fly_to_point(self, target_points, max_steps=1000, average_action=0):
        self.eval_env.zero_reset()
        if self.render:
            self.eval_env.drone_render_object.set_target(target_points)

        # first target
        current_target_ind = 0
        # target trajectory
        line_start = self.eval_env._state[:3]

        state = self.eval_env._state
        stable = True
        drone_traj, divergences = [], []
        last_actions = np.zeros((10, 4))
        step = 0

        while len(drone_traj) < max_steps:
            current_target = target_points[current_target_ind]
            action = self.controller.predict_actions(state, current_target)

            if average_action:
                # make average action
                if step == 0:
                    last_actions = action.copy()
                else:
                    last_actions = np.roll(last_actions, -1, axis=0)
                    # rolling mean
                    weight = np.expand_dims(9 - np.arange(10), 1)
                    # np.ones((10, 1)) # for giving more weight to newer states

                    last_actions = (last_actions * weight +
                                    action) / (weight + 1)
                # print("actions", action)
                # print("last actions", last_actions)
                step += 1
                use_action = last_actions[0]
            else:
                use_action = action[0]

            # if self.render:
            #     np.set_printoptions(suppress=1, precision=3)
            #     print(action[0])
            #     print()
            state, stable = self.eval_env.step(
                use_action, thresh_stable=self.thresh_stable
            )
            if self.render:
                self.eval_env.render()
                time.sleep(.05)

            # project drone onto line and compute divergence
            drone_on_line = project_to_line(
                line_start, current_target, state[:3]
            )
            div = np.linalg.norm(drone_on_line - state[:3])
            divergences.append(div)
            drone_traj.append(np.concatenate((state, action[0])))

            # set next target if we have passed one
            if state[0] > current_target[0]:
                if self.render:
                    np.set_printoptions(suppress=1, precision=3)
                    print("target:", current_target, "pos:", state[:3])
                if current_target_ind < len(target_points) - 1:
                    current_target_ind += 1
                    line_start = state[:3]
                else:
                    break

            if not stable or div > self.thresh_div:
                if self.render:
                    print("diverged", div, "stable", stable)
                    break
                else:
                    reset_state = np.zeros(12)
                    reset_state[:3] = drone_on_line
                    vec = current_target - drone_on_line
                    reset_state[3:6
                                ] = vec / np.linalg.norm(vec) * self.des_speed
                    self.eval_env._state = reset_state
        return np.array(drone_traj), np.array(divergences)

    def run_eval(self, nr_test, return_dists=False):
        mean_div, not_div_time = [], []
        for i in range(nr_test):
            target_point = [
                np.random.rand(3) * np.array([60, 10, 10]) +
                np.array([30, -5, -5])
            ]
            # traj_test = run_wing_flight(
            #     self.eval_env, traj_len=300, dt=self.dt, render=0
            # )
            # target_point = [traj_test[-1, :3]]
            drone_traj, divergences = self.fly_to_point(target_point)
            # last_x_points = drone_traj[-20:, :3]
            # last_x_dists = [
            #     np.linalg.norm(target_point - p) for p in last_x_points
            # ]
            # min_dists.append(np.min(last_x_dists))
            not_diverged = np.sum(divergences < self.thresh_div)
            mean_div.append(np.mean(divergences))
            not_div_time.append(not_diverged)
        mean_err = np.mean(mean_div)
        std_err = np.std(mean_div)
        print(
            "Time not diverged: %3.2f (%3.2f)" %
            (np.mean(not_div_time), np.std(not_div_time))
        )
        print("Average error: %3.2f (%3.2f)" % (mean_err, std_err))
        if return_dists:
            return np.array(mean_div)
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
        controller = MPC(horizon, dt, dynamics="fixed_wing_3D")
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
    params = {
        "render": 1,
        "dt": 0.05,
        "horizon": 10,
        "thresh_stable": 1,
        "thresh_div": 10
    }

    # load model
    model_name = args.model
    model_path = os.path.join("trained_models", "wing", model_name)

    controller = load_model(
        model_path, epoch=args.epoch, name="model_wing", **params
    )

    evaluator = FixedWingEvaluator(controller, **params)

    # only run evaluation without render
    # tic = time.time()
    # out_path = "../presentations/analysis"
    # evaluator.render = 0
    # dists_from_target = evaluator.run_eval(nr_test=10, return_dists=True)
    # np.save(os.path.join(out_path, "dists_mpc_last.npy"), dists_from_target)
    # print("time for 100 trajectories", time.time() - tic)
    # exit()

    target_point = [[50, -3, -3], [100, 3, 3]]

    # RUN
    drone_traj, _ = evaluator.fly_to_point(target_point, max_steps=1000)

    np.set_printoptions(suppress=True, precision=3)
    print("\n final state", drone_traj[-1])
    print(drone_traj.shape)
    # np.save(os.path.join(model_path, "drone_traj.npy"), drone_traj)

    evaluator.eval_env.close()

    # EVAL
    plot_wing_pos_3d(
        drone_traj,
        target_point,
        save_path=os.path.join(model_path, "coords.png")
    )
