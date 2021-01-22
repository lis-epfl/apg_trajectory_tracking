import os
import time
import argparse
import json
import numpy as np
import torch
import pickle

from environments.drone_env import QuadRotorEnvBase, trajectory_training_data
from utils.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist
)
from utils.trajectory import (
    sample_points_on_straight, sample_to_input, np_project_line,
    eval_get_straight_ref, get_reference, Circle
)
from dataset import DroneDataset
# from models.resnet_like_model import Net
from drone_loss import drone_loss_function

ROLL_OUT = 1
ACTION_DIM = 4

# Use cuda if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadEvaluator():

    def __init__(
        self,
        model,
        dataset,
        horizon=5,
        max_drone_dist=0.1,
        render=0,
        **kwargs
    ):
        self.dataset = dataset
        self.net = model
        self.eval_env = QuadRotorEnvBase()
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.training_means = None
        self.render = render
        self.treshold_divergence = 1

    def predict_actions(self, current_np_state, ref_states):
        """
        Predict an action for the current state. This function is used by all
        evaluation functions
        """
        # print([round(s, 2) for s in current_np_state])
        in_state, _, ref_body = self.dataset.get_and_add_eval_data(
            current_np_state, ref_states
        )
        # if self.render:
        #     self.check_ood(current_np_state, ref_world)

        # print([round(s, 2) for s in current_torch_state[0].numpy()])
        # print("reference", reference)
        # print("position", current_torch_state[:, 3:])
        suggested_action = self.net(in_state[:, 3:], ref_body)

        suggested_action = torch.sigmoid(suggested_action)[0]

        suggested_action = torch.reshape(suggested_action, (-1, ACTION_DIM))
        numpy_action_seq = suggested_action.cpu().numpy()
        # print([round(a, 2) for a in numpy_action_seq[0]])
        return numpy_action_seq

    def check_ood(self, drone_state, ref_states):
        if self.training_means is None:
            _, reference_training_data = trajectory_training_data(
                500, max_drone_dist=self.max_drone_dist
            )
            self.training_means = np.mean(reference_training_data, axis=0)
            self.training_std = np.std(reference_training_data, axis=0)
        drone_state_names = np.array(
            [
                "pos_x", "pos_y", "pos_z", "att_1", "att_2", "att_3", "vel_x",
                "vel_y", "vel_z", "rot_1", "rot_2", "rot_3", "rot_4",
                "att_vel_1", "att_vel_2", "att_vel_3"
            ]
        )
        ref_state_names = np.array(
            [
                "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "acc_1",
                "acc_2", "acc_3"
            ]
        )
        normed_drone_state = np.absolute(
            (drone_state - self.dataset.mean) / self.dataset.std
        )
        normed_ref_state = np.absolute(
            (ref_states - self.training_means) / self.training_std
        )
        if np.any(normed_drone_state > 3):
            print("state outlier:", drone_state_names[normed_drone_state > 3])
        if np.any(normed_ref_state > 3):
            for i in range(ref_states.shape[0]):
                print(
                    f"ref outlier (t={i}):",
                    ref_state_names[normed_ref_state[i] > 3]
                )

    def help_render(self, sleep=.05):
        """
        Helper function to make rendering prettier
        """
        if self.render:
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

    def circle_traj(self, max_nr_steps=200, plane=[0, 1], radius=1):
        # reset drone state
        self.eval_env.reset(strength=.1)
        current_np_state = self.eval_env._state.as_np

        # init circle
        circ_ref = Circle(plane=plane, radius=radius)
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
                self.help_render(sleep=0)

                # project to trajectory and check divergence
                drone_on_line = circ_ref.project_helper(drone_pos)
                reference_trajectory.append(drone_on_line)
                div = np.linalg.norm(drone_on_line - drone_pos)
                if div > self.treshold_divergence:
                    if self.render:
                        print("divregence to high", div)
                    break

        if self.render:
            self.eval_env.close()
            print(f"Circle: Steps until divergence: {i}")
            return np.array(reference_trajectory), drone_trajectory
        else:
            return i

    def straight_traj(self, max_nr_steps=200):
        # init_state = np.random.rand(3)
        # self.eval_env.zero_reset(*tuple(init_state))
        self.eval_env.reset()

        current_np_state = self.eval_env._state.as_np
        traj_direction = current_np_state[6:9]  # np.random.rand(3)
        a_on_line = current_np_state[:3]
        b_on_line = a_on_line + traj_direction / np.linalg.norm(traj_direction)

        drone_trajectory = []
        reference_trajectory = []  # drone states projected to ref
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

                current_np_state, stable = self.eval_env.step(action)
                drone_trajectory.append(current_np_state[:3])
                if not stable:
                    break

                self.help_render(sleep=0)

                drone_on_line = np_project_line(
                    a_on_line, b_on_line, current_np_state[:3]
                )
                reference_trajectory.append(drone_on_line)
                div = np.linalg.norm(drone_on_line - current_np_state[:3])
                if div > self.treshold_divergence:
                    if self.render:
                        print("divregence to high", div)
                    break
        # output dependent on render or not
        if self.render:
            return np.array(reference_trajectory), drone_trajectory
            self.eval_env.close()
        else:
            traj_len = np.linalg.norm(
                reference_trajectory[-1] - reference_trajectory[0]
            )
            steps_until_fail = i
            return traj_len, steps_until_fail

    def eval_ref(
        self, nr_test_straight=10, nr_test_circle=10, max_nr_steps=200
    ):
        """
        Function to evaluate both on straight and on circular traj
        """
        # ================ Straight ========================
        traj_len_stable, divergence = [], []
        for _ in range(nr_test_straight):
            traj_len, steps_until_fail = self.straight_traj(
                max_nr_steps=max_nr_steps
            )
            traj_len_stable.append(traj_len)
            divergence.append(steps_until_fail)

        # Output results for straight trajectory
        print(
            "Straight: Average trajectory length: %3.2f (%3.2f)" %
            (np.mean(traj_len_stable), np.std(traj_len_stable))
        )
        print(
            "Straight: Steps until divergence: %3.2f (%3.2f)" %
            (np.mean(divergence), np.std(divergence))
        )

        # ================= CIRCLE =======================
        circle_stable = []
        for _ in range(nr_test_circle):
            # vary plane and radius
            possible_planes = [[0, 1], [0, 2], [1, 2]]
            plane = possible_planes[np.random.randint(0, 3, 1)[0]]
            radius = np.random.rand() + .5  # at least .5, max 1.5
            # run
            steps_until_div = self.circle_traj(
                max_nr_steps=max_nr_steps, plane=plane, radius=radius
            )
            circle_stable.append(steps_until_div)

        # output results for circle:
        print(
            "Circle: Steps until divergence: %3.2f (%3.2f)" %
            (np.mean(circle_stable), np.std(circle_stable))
        )
        return np.mean(traj_len_stable), np.std(traj_len_stable)


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

    dataset = DroneDataset(num_states=1, **param_dict)
    evaluator = QuadEvaluator(net, dataset, render=1, **param_dict)

    # Straight with reference as input
    try:
        # STRAIGHT
        initial_trajectory, drone_trajectory = evaluator.straight_traj(
            max_nr_steps=300,
        )

        plot_trajectory(
            initial_trajectory, drone_trajectory,
            os.path.join(model_path, "traj.png")
        )

        # CIRCLE
        # ref_trajectory, drone_trajectory = evaluator.circle_traj(
        #     max_nr_steps=1000
        # )
        # plot_trajectory(
        #     ref_trajectory, drone_trajectory,
        #     os.path.join(model_path, "circle_traj.png")
        # )

        # MEASURE change by drone dist
        # success_mean_list = []
        # distances = np.arange(0.5, 2.3, 0.2) * param_dict["max_drone_dist"]
        # for drone_dist in distances:
        #     evaluator.max_drone_dist = drone_dist
        #     suc_mean, suc_std = evaluator.eval_traj_input(
        #         nr_test_data=20,
        #         max_nr_steps=200,
        #     )
        #     success_mean_list.append(suc_mean)

        # plot_suc_by_dist(distances, success_mean_list, model_path)

        # STABILIZE
        # drone_trajectory = evaluator.stabilize(
        #     nr_test_data=1,
        #     max_nr_steps=300,
        # )
        # plot_position(drone_trajectory, os.path.join(model_path, "stable.png"))

        # evaluator.max_drone_dist = param_dict["max_drone_dist"] * 2
        # evaluator.eval_traj_input(threshold_divergence, nr_test_data=20)
        # max_nr_steps=200, step_size=param_dict["step_size"])
    except KeyboardInterrupt:
        evaluator.eval_env.close()
    # evaluator.evaluate()
