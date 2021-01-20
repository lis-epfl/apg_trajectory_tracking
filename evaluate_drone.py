import os
import time
import argparse
import json
import numpy as np
import torch
import pickle

from environments.drone_env import QuadRotorEnvBase
from utils.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist
)
from utils.trajectory import (
    sample_points_on_straight, sample_to_input, np_project_line,
    eval_get_straight_ref, get_reference, Circle
)
from dataset import raw_states_to_torch
# from models.resnet_like_model import Net
from drone_loss import drone_loss_function

ROLL_OUT = 1
ACTION_DIM = 4

# Use cuda if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadEvaluator():

    def __init__(
        self, model, mean=0, std=1, horizon=5, max_drone_dist=0.1, **kwargs
    ):
        self.mean = mean
        # self.mean[2] -= 2  # for old models
        self.std = std
        self.net = model
        self.eval_env = QuadRotorEnvBase()
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist

    def predict_actions(self, current_np_state, ref_states=None):
        """
        Predict an action for the current state. This function is used by all
        evaluation functions
        """
        # print([round(s, 2) for s in current_np_state])
        current_torch_state = raw_states_to_torch(
            current_np_state, normalize=True, mean=self.mean, std=self.std
        ).to(device)
        if ref_states is not None:
            ref_states[:, :3] = ref_states[:, :3] - current_np_state[:3]
            reference = torch.unsqueeze(
                torch.from_numpy(ref_states).float(), 0
            )
            # print([round(s, 2) for s in current_torch_state[0].numpy()])
            # print("reference", reference)
            # print("position", current_torch_state[:, 3:])
            suggested_action = self.net(current_torch_state[:, 3:], reference)
        else:
            suggested_action = self.net(current_torch_state)
        suggested_action = torch.sigmoid(suggested_action)[0]

        suggested_action = torch.reshape(suggested_action, (-1, ACTION_DIM))
        numpy_action_seq = suggested_action.cpu().numpy()
        # print([round(a, 2) for a in numpy_action_seq[0]])
        return numpy_action_seq

    def help_render(self, render, sleep=.05):
        """
        Helper function to make rendering prettier
        """
        if render:
            # print([round(s, 2) for s in current_np_state])
            current_np_state = self.eval_env._state.as_np
            self.eval_env._state.set_position(
                current_np_state[:3] + np.array([0, 0, 1])
            )
            self.eval_env.render()
            self.eval_env._state.set_position(current_np_state[:3])
            time.sleep(sleep)

    def stabilize(self, nr_test_data=1, max_nr_steps=300, render=True):
        nr_stable = []
        for k in range(nr_test_data):
            self.eval_env.reset()
            current_np_state = self.eval_env._state.as_np
            # Goal state: constant position and velocity
            posf = current_np_state[:3].copy()
            # if render:
            print("target pos:", posf)
            velf = np.zeros(3)
            # Run
            drone_trajectory = []
            with torch.no_grad():
                for i in range(max_nr_steps):
                    # goal state: same pos, zero velocity
                    pos0 = current_np_state[:3]
                    vel0 = current_np_state[6:9]
                    acc0 = self.eval_env.get_acceleration()
                    trajectory = get_reference(
                        pos0,
                        vel0,
                        acc0,
                        posf,
                        velf,
                        delta_t=self.eval_env.dt,
                        ref_length=self.horizon
                    )
                    numpy_action_seq = self.predict_actions(
                        current_np_state, trajectory
                    )
                    action = numpy_action_seq[0]
                    current_np_state, stable = self.eval_env.step(action)
                    drone_trajectory.append(current_np_state[:3])
                    if not stable:
                        break
                    self.help_render(render)
            nr_stable.append(i)
        print("Average episode length: ", np.mean(nr_stable))
        return drone_trajectory

    def circle_traj(self, thresh, max_nr_steps=200, render=False):
        # reset drone state
        self.eval_env.reset(strength=.1)
        current_np_state = self.eval_env._state.as_np

        # init circle
        circ_ref = Circle(plane=[0, 1], radius=1)
        circ_ref.init_from_tangent(current_np_state[:3], current_np_state[6:9])

        reference_trajectory = []
        drone_trajectory = []
        with torch.no_grad():
            for i in range(max_nr_steps):
                acc = self.eval_env.get_acceleration()
                trajectory = circ_ref.eval_get_circle(
                    current_np_state, acc, self.max_drone_dist, self.horizon
                )
                numpy_action_seq = self.predict_actions(
                    current_np_state, trajectory
                )
                # only use first action (as in mpc)
                action = numpy_action_seq[0]
                current_np_state, stable = self.eval_env.step(action)
                drone_pos = current_np_state[:3]
                drone_trajectory.append(drone_pos)
                if not stable:
                    break
                self.help_render(render, sleep=0)

                # project to trajectory and check divergence
                drone_on_line = circ_ref.project_helper(drone_pos)
                reference_trajectory.append(drone_on_line)
                div = np.linalg.norm(drone_on_line - drone_pos)
                if div > thresh:
                    if render:
                        print("divregence to high", div)
                    break

        print(f"Number of steps until divergence / failure {i}")
        self.eval_env.close()
        return np.array(reference_trajectory), drone_trajectory

    def eval_traj_input(
        self, thresh, nr_test_data=5, max_nr_steps=200, render=False
    ):
        nr_stable, divergence = [], []
        for k in range(nr_test_data):
            # init_state = np.random.rand(3)
            # self.eval_env.zero_reset(*tuple(init_state))
            self.eval_env.reset()

            current_np_state = self.eval_env._state.as_np
            traj_direction = current_np_state[6:9]  # np.random.rand(3)
            a_on_line = current_np_state[:3]
            b_on_line = a_on_line + traj_direction / np.linalg.norm(
                traj_direction
            )

            # trajectory = sample_points_on_straight(current_np_state[:3],
            # traj_direction, step_size=step_size, ref_length=self.horizon)
            initial_trajectory = [a_on_line]
            # if the reference is input relative to drone state,
            # there is no need to roll?
            # actually there is, because change of drone state

            drone_trajectory = []
            with torch.no_grad():
                for i in range(max_nr_steps):
                    acc = self.eval_env.get_acceleration()
                    trajectory = eval_get_straight_ref(
                        current_np_state, acc, a_on_line, b_on_line,
                        self.max_drone_dist, self.horizon
                    )
                    # ref_s = sample_to_input(current_np_state, trajectory)
                    numpy_action_seq = self.predict_actions(
                        current_np_state, trajectory
                    )
                    # only use first action (as in mpc)
                    action = numpy_action_seq[0]
                    # for nr_action in range(ROLL_OUT):
                    #     # retrieve next action
                    #     action = numpy_action_seq[nr_action]
                    #     # take step in environment
                    # for action in numpy_action_seq:
                    # print(action)
                    current_np_state, stable = self.eval_env.step(action)
                    drone_trajectory.append(current_np_state[:3])
                    if not stable:
                        break
                    # # update reference - 1) next timestep:
                    # trajectory = np.roll(trajectory, -1, axis=0)
                    # trajectory[-1] = 2* trajectory[-2] - trajectory[-3]
                    # initial_trajectory.append(trajectory[0, :3])
                    # # 2) project pos to line
                    # new_point_on_line = np_project_line(trajectory[0],
                    #  trajectory[1], current_np_state[:3])
                    # trajectory = sample_points_on_straight(new_point_on_line,
                    #  traj_direction, step_size=step_size)
                    self.help_render(render)

                    drone_on_line = np_project_line(
                        a_on_line, b_on_line, current_np_state[:3]
                    )
                    initial_trajectory.append(drone_on_line)
                    div = np.linalg.norm(drone_on_line - current_np_state[:3])
                    if div > thresh:
                        if render:
                            print("divregence to high", div)
                        break
            nr_stable.append(i)
            divergence.append(i)
        # print("Number of stable steps:", round(np.mean(nr_stable)))
        print(
            f"Number of steps until divergence {round(np.mean(divergence), 2)}\
                 ({round(np.std(divergence), 2)})"
        )
        self.eval_env.close()
        # print(initial_trajectory)
        if render:
            return np.array(initial_trajectory), drone_trajectory
        else:
            return round(np.mean(divergence), 2), round(np.std(divergence), 2)


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
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join("trained_models", "drone", model_name)

    # load std or other parameters from json
    with open(os.path.join(model_path, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, "model_quad" + args.epoch))
    net = net.to(device)
    net.eval()

    evaluator = QuadEvaluator(net, **param_dict)

    threshold_divergence = 5 * param_dict["max_drone_dist"]
    # Straight with reference as input
    try:
        # STRAIGHT
        # initial_trajectory, drone_trajectory = evaluator.eval_traj_input(
        #     threshold_divergence,
        #     nr_test_data=1,
        #     render=True,
        #     max_nr_steps=300,
        # )
        # plot_trajectory(
        #     initial_trajectory, drone_trajectory,
        #     os.path.join(model_path, "traj.png")
        # )

        # CIRCLE
        ref_trajectory, drone_trajectory = evaluator.circle_traj(
            threshold_divergence, max_nr_steps=1000, render=1
        )
        plot_trajectory(
            ref_trajectory, drone_trajectory,
            os.path.join(model_path, "circle_traj.png")
        )

        # MEASURE change by drone dist
        # success_mean_list = []
        # distances = np.arange(0.5, 2.3, 0.2) * param_dict["max_drone_dist"]
        # for drone_dist in distances:
        #     evaluator.max_drone_dist = drone_dist
        #     suc_mean, suc_std = evaluator.eval_traj_input(
        #         threshold_divergence,
        #         nr_test_data=20,
        #         max_nr_steps=200,
        #     )
        #     success_mean_list.append(suc_mean)

        # plot_suc_by_dist(distances, success_mean_list, model_path)

        # STABILIZE
        # drone_trajectory = evaluator.stabilize(
        #     nr_test_data=1,
        #     render=False,
        #     max_nr_steps=300,
        # )
        # plot_position(drone_trajectory, os.path.join(model_path, "stable.png"))

        # evaluator.max_drone_dist = param_dict["max_drone_dist"] * 2
        # evaluator.eval_traj_input(threshold_divergence, nr_test_data=20)
        # render=False, max_nr_steps=200, step_size=param_dict["step_size"])
    except KeyboardInterrupt:
        evaluator.eval_env.close()
    # evaluator.evaluate()
