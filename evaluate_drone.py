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
    eval_get_straight_ref, get_reference
)
from utils.circle import Circle
from environments.rendering import CircleObject, StraightObject
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
            current_np_state, ref_states
        )
        # if self.render:
        #     self.check_ood(current_np_state, ref_world)
        # print(current_np_state)
        # print(in_state)
        # print()
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
                    self.help_render(sleep=0)
            nr_stable.append(i)
        print("Average episode length: ", np.mean(nr_stable))
        return drone_trajectory

    def circle_traj(self, max_nr_steps=200, thresh=.4, **circle_args):
        """
        Follow a circle with the drone environment
        """
        # reset drone state
        init_state = [0, 0, 3]
        self.eval_env.zero_reset(*tuple(init_state))

        states = None  # np.load("id_5.npy")
        # Option to load data
        if states is not None:
            self.eval_env._state.from_np(states[0])

        # get current state
        current_np_state = self.eval_env._state.as_np

        # init circle
        circ_ref = Circle(**circle_args)
        circ_ref.init_from_tangent(
            current_np_state[:3].copy(), current_np_state[6:9].copy()
        )

        if self.render:
            self.eval_env.renderer.add_object(
                CircleObject(circ_ref.mid_point, circle_args["radius"])
            )

        (reference_trajectory, drone_trajectory,
         divergences) = [], [current_np_state], []
        with torch.no_grad():
            for i in range(max_nr_steps):
                acc = self.eval_env.get_acceleration()
                trajectory = circ_ref.eval_get_circle(
                    current_np_state,
                    acc,
                    self.max_drone_dist,
                    self.horizon,
                    dt=self.dt
                )
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
                    break
                self.help_render(sleep=0)

                # project to trajectory and check divergence
                drone_on_line = circ_ref.project_helper(drone_pos)
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
            print(f"Circle: Steps until divergence: {i}")
            return np.array(reference_trajectory), drone_trajectory
        else:
            avg_diff = np.mean(divergences)
            return i, avg_diff

    def straight_traj(self, max_nr_steps=200):
        init_state = [2, 0, 3]
        self.eval_env.zero_reset(*tuple(init_state))

        current_np_state = self.eval_env._state.as_np
        traj_direction = current_np_state[6:9]  # np.random.rand(3)
        a_on_line = current_np_state[:3]
        b_on_line = a_on_line + 5 * traj_direction / np.linalg.norm(
            traj_direction
        )

        if self.render:
            self.eval_env.renderer.add_object(
                StraightObject(a_on_line, b_on_line)
            )

        drone_trajectory = [current_np_state]
        reference_trajectory = []  # drone states projected to ref
        divergences = []
        with torch.no_grad():
            for i in range(max_nr_steps):
                acc = self.eval_env.get_acceleration()
                trajectory = eval_get_straight_ref(
                    current_np_state, acc, a_on_line, b_on_line,
                    self.max_drone_dist, self.horizon, self.dt
                )
                # ref_s = sample_to_input(current_np_state, trajectory)
                numpy_action_seq = self.predict_actions(
                    current_np_state, trajectory
                )
                # only use first action (as in mpc)
                action = numpy_action_seq[0]

                current_np_state, stable = self.eval_env.step(action)
                drone_trajectory.append(current_np_state)
                if not stable:
                    break

                self.help_render(sleep=0)

                drone_on_line = np_project_line(
                    a_on_line, b_on_line, current_np_state[:3]
                )
                reference_trajectory.append(drone_on_line)
                div = np.linalg.norm(drone_on_line - current_np_state[:3])
                divergences.append(div)
                if div > self.treshold_divergence:
                    if self.render:
                        print("divergence too high", div)
                    break
        # output dependent on render or not
        # distance = np.linalg.norm(
        #     reference_trajectory[-1] - reference_trajectory[0]
        # )
        # print("Distance:", distance)
        # print("Speed:", distance / (i * self.dt))
        if self.render:
            return np.array(reference_trajectory), drone_trajectory
            self.eval_env.close()
        else:
            avg_div = np.mean(divergences)
            steps_until_fail = i
            return steps_until_fail, avg_div

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
            steps_until_fail, avg_div = self.straight_traj(
                max_nr_steps=max_steps_straight
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
            steps_until_div, avg_diff = self.circle_traj(
                max_nr_steps=max_steps_circle, plane=plane, radius=radius
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
    # evaluator.eval_ref(max_steps_circle=1000)
    # exit()
    # Straight with reference as input
    try:
        # STRAIGHT
        if args.ref == "straight":
            initial_trajectory, drone_trajectory = evaluator.straight_traj(
                max_nr_steps=1000,
            )
            plot_trajectory(
                initial_trajectory, drone_trajectory,
                os.path.join(model_path, "traj.png")
            )

        # evaluator.collect_training_data()

        # CIRCLE
        if args.ref == "circle":
            plane = [0, 2]
            fixed_axis = 1
            ref_trajectory, drone_trajectory = evaluator.circle_traj(
                max_nr_steps=1000,
                radius=2,
                plane=plane,
                thresh=1,
                direction=1
            )
            plot_trajectory(
                ref_trajectory,
                drone_trajectory,
                os.path.join(model_path, "circle_traj.png"),
                fixed_axis=fixed_axis
            )
        # # print(drone_trajectory[0, :3])
        # # print(drone_trajectory[-1, :3])
        # np.save("euler.npy", drone_trajectory)

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
        if args.ref == "stable":
            drone_trajectory = evaluator.stabilize(
                nr_test_data=1,
                max_nr_steps=1000,
            )
            plot_position(
                drone_trajectory, os.path.join(model_path, "stable.png")
            )

        # evaluator.max_drone_dist = param_dict["max_drone_dist"] * 2
        # evaluator.eval_traj_input(threshold_divergence, nr_test_data=20)
        # max_nr_steps=200, step_size=param_dict["step_size"])
    except KeyboardInterrupt:
        evaluator.eval_env.close()
    # evaluator.evaluate()
