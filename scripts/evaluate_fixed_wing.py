import os
import time
import argparse
import json
import numpy as np
import torch

from neural_control.environments.wing_env import SimpleWingEnv, run_wing_flight
from neural_control.plotting import plot_wing_pos_3d, plot_success
from neural_control.dataset import WingDataset
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.controllers.mpc import MPC
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.trajectory.q_funcs import project_to_line
from evaluate_base import run_mpc_analysis, load_model_params, average_action
from neural_control.environments.rendering import animate_fixed_wing


class FixedWingEvaluator:
    """
    Evaluate performance of the fixed wing drone
    """

    def __init__(
        self,
        controller,
        env,
        dt=0.01,
        horizon=1,
        render=0,
        thresh_div=10,
        thresh_stable=0.8,
        test_time=0,
        **kwargs
    ):
        self.controller = controller
        self.dt = dt
        self.horizon = horizon
        self.render = render
        self.thresh_div = thresh_div
        self.thresh_stable = thresh_stable
        self.eval_env = env
        self.des_speed = 11.5
        self.test_time = test_time

    def fly_to_point(
        self, target_points, max_steps=1000, do_avg_act=0, return_traj=False
    ):
        self.eval_env.zero_reset()
        if self.render:
            self.eval_env.drone_render_object.set_target(target_points)

        # first target
        current_target_ind = 0
        # target trajectory
        line_start = self.eval_env._state[:3]

        state = self.eval_env._state
        stable = True

        drone_traj, div_to_linear, div_target = [], [], []
        step = 0
        while len(drone_traj) < max_steps:
            current_target = target_points[current_target_ind]
            action = self.controller.predict_actions(state, current_target)

            use_action = action[0]

            step += 1

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
            div_to_linear.append(div)
            drone_traj.append(np.concatenate((state, action[0])))

            # set next target if we have passed one
            if state[0] > current_target[0]:
                # project target onto line
                target_on_traj = project_to_line(
                    drone_traj[-2][:3], state[:3], current_target
                )
                div_target.append(
                    np.linalg.norm(target_on_traj - current_target)
                )
                if self.render:
                    np.set_printoptions(suppress=1, precision=3)
                    print(
                        "target:", current_target, "pos:", state[:3],
                        "div to target", div_target[-1]
                    )
                if current_target_ind < len(target_points) - 1:
                    current_target_ind += 1
                    line_start = state[:3]
                else:
                    break

            if not stable or div > self.thresh_div:
                div_target.append(self.thresh_div)
                if self.test_time:
                    div_target[-1] = np.linalg.norm(state[:3] - current_target)
                    # print("diverged", div, "stable", stable)
                    break
                else:
                    reset_state = np.zeros(12)
                    reset_state[:3] = drone_on_line
                    vec = current_target - drone_on_line
                    reset_state[3:6
                                ] = vec / np.linalg.norm(vec) * self.des_speed
                    self.eval_env._state = reset_state
        if len(drone_traj) == max_steps:
            print("Reached max steps")
            div_target.append(self.thresh_div)
        if return_traj:
            return np.array(drone_traj)
        else:
            return np.array(div_target), np.array(div_to_linear)

    def run_eval(
        self, nr_test, return_dists=False, x_dist=50, x_std=5, printout=True
    ):
        self.dyn_eval_test = []
        mean_div_target, mean_div_linear, not_div_time = [], [], []
        for i in range(nr_test):
            # important! reset after every run
            if isinstance(self.controller, MPC):
                self.controller._initDynamics()
            # set target point
            rand_y, rand_z = tuple((np.random.rand(2) - .5) * 2 * x_std)
            # TODO: MAIN CHANGE
            target_point = np.array([[x_dist, rand_y, rand_z]])
            # target_point = [
            #     np.random.rand(3) * np.array([70, 10, 10]) +
            #     np.array([20, -5, -5])
            # ]

            # for overfitting
            # target_point = np.array([[x_dist, -3, 3]])
            # self.eval_env.dynamics.timestamp = np.pi / 2
            div_target, div_linear = self.fly_to_point(target_point)

            mean_div_target.append(np.mean(div_target))
            mean_div_linear.append(np.mean(div_linear))
            not_div_time.append(len(div_linear))

        mean_err_target, std_err_target = (
            np.mean(mean_div_target), np.std(mean_div_target)
        )

        if printout:
            # print(
            #     "Average error (traj): %3.2f (%3.2f)" %
            #     (np.mean(mean_div_linear), np.std(mean_div_linear))
            # )
            print(
                "Time not diverged: %3.2f (%3.2f)" %
                (np.mean(not_div_time), np.std(not_div_time))
            )
            print(
                "Average error (target): %3.2f (%3.2f)" %
                (mean_err_target, std_err_target)
            )
        if return_dists:
            return np.array(mean_div_target)
        return mean_err_target, std_err_target


def load_model(model_path, epoch="", **kwargs):
    """
    Load model and corresponding parameters
    """
    net, param_dict = load_model_params(model_path, "model_wing", epoch=epoch)
    dataset = WingDataset(0, **param_dict)

    controller = FixedWingNetWrapper(net, dataset, **param_dict)
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
        "-n", "--animate", action='store_true', help="animate 3D"
    )
    parser.add_argument(
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    parser.add_argument(
        "-a", "--eval", type=int, default=0, help="number eval runs"
    )
    parser.add_argument(
        "-s", "--save_traj", action='store_true', help="number eval runs"
    )
    args = parser.parse_args()

    # parameters
    params = {
        "render": 0,
        "dt": 0.05,
        "horizon": 10,
        "thresh_stable": 3,
        "thresh_div": 10
    }

    # load model
    model_name = args.model
    model_path = os.path.join("trained_models", "wing", model_name)

    if model_name != "mpc":
        controller = load_model(
            model_path, epoch=args.epoch, name="model_wing", **params
        )
    else:
        controller = MPC(20, 0.1, dynamics="fixed_wing_3D")

    modified_params = {}

    dynamics = FixedWingDynamics(modified_params=modified_params)
    eval_env = SimpleWingEnv(dynamics, params["dt"])
    evaluator = FixedWingEvaluator(controller, eval_env, test_time=1, **params)

    # only run evaluation without render
    if args.eval > 0:
        # tic = time.time()
        out_path = "../presentations/analysis"
        evaluator.run_eval(nr_test=args.eval, return_dists=True)
        exit()

    target_point = [[50, 6, -4]]

    # Run (one trial)
    drone_traj = evaluator.fly_to_point(
        target_point, max_steps=1000, return_traj=True
    )
    if args.save_traj:
        os.makedirs("output_video", exist_ok=True)
        np.save(
            os.path.join("output_video", f"wing_traj_{args.model}.npy"),
            drone_traj
        )
    if args.animate:
        animate_fixed_wing(
            target_point, [drone_traj]
            # uncomment to save video
            # savefile=os.path.join(model_path, 'video.mp4')
        )

    evaluator.eval_env.close()
