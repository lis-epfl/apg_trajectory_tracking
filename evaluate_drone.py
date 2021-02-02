import os
import time
import argparse
import json
import numpy as np
import torch
import pickle

from environments.drone_env import QuadRotorEnvBase, trajectory_training_data
from utils.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist,
    plot_drone_ref_coords
)
from utils.trajectory import Hover, Straight
from utils.circle import Circle
from utils.random_reference import RandomReference
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
        dt=0.02,
        **kwargs
    ):
        self.dataset = dataset
        self.net = model
        self.eval_env = QuadRotorEnvBase(dt)
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.training_means = None
        self.render = render
        self.treshold_divergence = 1
        self.dt = dt

    def predict_actions(self, current_np_state, ref_states):
        """
        Predict an action for the current state. This function is used by all
        evaluation functions
        """
        # print([round(s, 2) for s in current_np_state])
        in_state, _, ref_body = self.dataset.get_and_add_eval_data(
            current_np_state.copy(), ref_states
        )
        # if self.render:
        #     self.check_ood(current_np_state, ref_world)
        # np.set_printoptions(suppress=True, precision=0)
        # print(current_np_state)
        suggested_action = self.net(in_state, ref_body)

        suggested_action = torch.sigmoid(suggested_action)[0]

        suggested_action = torch.reshape(suggested_action, (-1, ACTION_DIM))
        numpy_action_seq = suggested_action.cpu().numpy()
        # print([round(a, 2) for a in numpy_action_seq[0]])
        return numpy_action_seq

    def check_ood(self, drone_state, ref_states):
        if self.training_means is None:
            _, reference_training_data = trajectory_training_data(
                500, max_drone_dist=self.max_drone_dist, dt=self.dt
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

    def follow_trajectory(
        self, traj_type, max_nr_steps=200, thresh=.4, **circle_args
    ):
        """
        Follow a trajectory with the drone environment
        Argument trajectory: Can be any of
                straight
                circle
                hover
                poly
        """
        # reset drone state
        init_state = [3, 0, 6]
        self.eval_env.zero_reset(*tuple(init_state))

        states = None  # np.load("id_5.npy")
        # Option to load data
        if states is not None:
            self.eval_env._state.from_np(states[0])

        # get current state
        current_np_state = self.eval_env._state.as_np

        # Get right trajectory object:
        object_dict = {
            "hover": Hover,
            "straight": Straight,
            "circle": Circle,
            "poly": RandomReference
        }
        reference = object_dict[traj_type](
            current_np_state.copy(),
            self.render,
            self.eval_env.renderer,
            max_drone_dist=self.max_drone_dist,
            horizon=self.horizon,
            dt=self.dt,
            **circle_args
        )

        (reference_trajectory, drone_trajectory,
         divergences) = [], [current_np_state], []
        with torch.no_grad():
            for i in range(max_nr_steps):
                acc = self.eval_env.get_acceleration()
                trajectory = reference.get_ref_traj(current_np_state, acc)
                numpy_action_seq = self.predict_actions(
                    current_np_state, trajectory
                )
                # only use first action (as in mpc)
                action = numpy_action_seq[0]
                current_np_state, stable = self.eval_env.step(
                    action, thresh=thresh
                )
                if states is not None:
                    self.eval_env._state.from_np(states[i])
                    current_np_state = states[i]
                    stable = i < (len(states) - 1)
                drone_pos = current_np_state[:3]
                drone_trajectory.append(current_np_state)
                if not stable:
                    print("unstable")
                    break
                self.help_render(sleep=0)

                # project to trajectory and check divergence
                drone_on_line = reference.project_on_ref(drone_pos)
                reference_trajectory.append(trajectory[-1, :3])
                div = np.linalg.norm(drone_on_line - drone_pos)
                divergences.append(div)
                if div > self.treshold_divergence:
                    if self.render:
                        np.set_printoptions(precision=3, suppress=True)
                        print("state")
                        print([round(s, 2) for s in current_np_state])
                        print("trajectory:")
                        print(np.around(trajectory, 2))
                    break
        if self.render:
            self.eval_env.close()
            return np.array(reference_trajectory), np.array(drone_trajectory)
        else:
            avg_div = np.mean(divergences)
            return i, avg_div

    def eval_ref(
        self,
        nr_test_straight=10,
        nr_test_circle=10,
        max_steps_straight=200,
        max_steps_circle=200
    ):
        """
        Function to evaluate both on straight and on circular traj
        """
        # ================ Straight ========================
        straight_div, straight_stable = [], []
        for _ in range(nr_test_straight):
            steps_until_fail, avg_div = self.follow_trajectory(
                "straight", max_nr_steps=max_steps_straight
            )
            straight_div.append(avg_div)
            straight_stable.append(steps_until_fail)

        # Output results for straight trajectory
        print(
            "Straight: Average divergence: %3.2f (%3.2f)" %
            (np.mean(straight_div), np.std(straight_div))
        )
        print(
            "Straight: Steps until divergence: %3.2f (%3.2f)" %
            (np.mean(straight_stable), np.std(straight_stable))
        )

        # ================= CIRCLE =======================
        circle_div, circle_stable = [], []
        for _ in range(nr_test_circle):
            # vary plane and radius
            possible_planes = [[0, 1], [0, 2], [1, 2]]
            plane = possible_planes[np.random.randint(0, 3, 1)[0]]
            radius = np.random.rand() + .5  # at least .5, max 1.5
            # run
            steps_until_div, avg_diff = self.follow_trajectory(
                "circle",
                max_nr_steps=max_steps_circle,
                plane=plane,
                radius=radius
            )
            circle_stable.append(steps_until_div)
            circle_div.append(avg_diff)

        # output results for circle:
        print(
            "Circle: Average divergence: %3.2f (%3.2f)" %
            (np.mean(circle_div), np.std(circle_div))
        )
        print(
            "Circle: Steps until divergence: %3.2f (%3.2f)" %
            (np.mean(circle_stable), np.std(circle_stable))
        )
        return np.mean(circle_stable), np.std(circle_stable)

    def collect_training_data(self, outpath="data/jan_2021.npy"):
        """
        Run evaluation but collect and save states as training data
        """
        data = []
        for _ in range(80):
            _, drone_traj = self.straight_traj(max_nr_steps=100)
            data.extend(drone_traj)
        for _ in range(20):
            # vary plane and radius
            possible_planes = [[0, 1], [0, 2], [1, 2]]
            plane = possible_planes[np.random.randint(0, 3, 1)[0]]
            radius = np.random.rand() + .5
            # run
            _, drone_traj = self.circle_traj(
                max_nr_steps=500, radius=radius, plane=plane
            )
            data.extend(drone_traj)
        data = np.array(data)
        print(data.shape)
        np.save(outpath, data)


def load_model(model_path, epoch=""):
    """
    Load model and corresponding parameters
    """
    # load std or other parameters from json
    with open(os.path.join(model_path, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, "model_quad" + epoch))
    net = net.to(device)
    net.eval()
    return net, param_dict


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
        "-r", "--ref", type=str, default="circle", help="which trajectory"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    model_name = args.model
    model_path = os.path.join("trained_models", "drone", model_name)

    net, param_dict = load_model(model_path, epoch=args.epoch)

    dataset = DroneDataset(num_states=1, **param_dict)
    evaluator = QuadEvaluator(net, dataset, render=1, **param_dict)
    # evaluator.eval_ref()
    # exit()
    # Straight with reference as input
    try:
        fixed_axis = 1
        circle_args = {
            "plane": [0, 2],
            "radius": 2,
            "direction": -1,
            "thresh": 1
        }
        reference_traj, drone_traj = evaluator.follow_trajectory(
            args.ref, max_nr_steps=1000, **circle_args
        )
        plot_trajectory(
            reference_traj,
            drone_traj,
            os.path.join(model_path, args.ref + "_traj.png"),
            fixed_axis=fixed_axis
        )
        plot_drone_ref_coords(
            drone_traj[1:, :3], reference_traj,
            os.path.join(model_path, args.ref + "_coords.png")
        )

    except KeyboardInterrupt:
        evaluator.eval_env.close()
    # evaluator.evaluate()
