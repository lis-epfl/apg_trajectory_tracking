import os
import time
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import pickle
import contextlib

from neural_control.environments.drone_env import (
    QuadRotorEnvBase, trajectory_training_data
)
from neural_control.utils.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist,
    plot_drone_ref_coords
)
from neural_control.utils.straight import Hover, Straight
from neural_control.utils.circle import Circle
from neural_control.utils.polynomial import Polynomial
from neural_control.dataset import DroneDataset
from neural_control.drone_loss import reference_loss
from neural_control.environments.drone_dynamics import simulate_quadrotor
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

ROLL_OUT = 1
ACTION_DIM = 4

# Use cuda if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextlib.contextmanager
def dummy_context():
    yield None


class QuadEvaluator():

    def __init__(
        self,
        model,
        dataset,
        optimizer=None,
        horizon=5,
        max_drone_dist=0.1,
        render=0,
        dt=0.02,
        take_every_x=1000,
        **kwargs
    ):
        self.dataset = dataset
        self.net = model
        self.eval_env = QuadRotorEnvBase(dt)
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.training_means = None
        self.render = render
        self.dt = dt
        self.optimizer = optimizer
        self.take_every_x = take_every_x
        self.action_counter = 0

    def predict_actions(self, current_np_state, ref_states):
        """
        Predict an action for the current state. This function is used by all
        evaluation functions
        """
        # determine whether we also add the sample to our train data
        add_to_dataset = (self.action_counter + 1) % self.take_every_x == 0
        # preprocess state
        in_state, current_state, ref = self.dataset.get_and_add_eval_data(
            current_np_state.copy(), ref_states, add_to_dataset=add_to_dataset
        )
        # check if we want to train on this sample
        do_training = (
            (self.optimizer is not None)
            and np.random.rand() < 1 / self.take_every_x
        )
        with dummy_context() if do_training else torch.no_grad():
            # if self.render:
            #     self.check_ood(current_np_state, ref_world)
            # np.set_printoptions(suppress=True, precision=0)
            # print(current_np_state)
            suggested_action = self.net(in_state, ref)

            suggested_action = torch.sigmoid(suggested_action)[0]

            suggested_action = torch.reshape(
                # batch size 1
                suggested_action,
                (1, self.horizon, ACTION_DIM)
            )

        if do_training:
            self.optimizer.zero_grad()

            intermediate_states = torch.zeros(
                in_state.size()[0], self.horizon,
                current_state.size()[1]
            )
            for k in range(self.horizon):
                # extract action
                action = suggested_action[:, k]
                current_state = simulate_quadrotor(
                    action, current_state, dt=self.dt
                )
                intermediate_states[:, k] = current_state

            # print(intermediate_states.size(), ref.size())
            loss = reference_loss(
                intermediate_states, ref, printout=0, delta_t=self.dt
            )

            # Backprop
            loss.backward()
            self.optimizer.step()

        numpy_action_seq = suggested_action[0].detach().numpy()
        # print([round(a, 2) for a in numpy_action_seq[0]])
        # keep track of actions
        self.action_counter += 1
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
        self,
        traj_type,
        max_nr_steps=200,
        thresh_stable=.4,
        thresh_div=3,
        **traj_args
    ):
        """
        Follow a trajectory with the drone environment
        Argument trajectory: Can be any of
                straight
                circle
                hover
                poly
        """
        # reset action counter for new trajectory
        self.action_counter = 0

        # reset drone state
        init_state = [0, 0, 3]
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
            "poly": Polynomial
        }
        reference = object_dict[traj_type](
            current_np_state.copy(),
            self.render,
            self.eval_env.renderer,
            max_drone_dist=self.max_drone_dist,
            horizon=self.horizon,
            dt=self.dt,
            **traj_args
        )

        self.help_render()
        # start = input("start")

        (reference_trajectory, drone_trajectory,
         divergences) = [], [current_np_state], []
        for i in range(max_nr_steps):
            acc = self.eval_env.get_acceleration()
            trajectory = reference.get_ref_traj(current_np_state, acc)
            numpy_action_seq = self.predict_actions(
                current_np_state, trajectory
            )
            # only use first action (as in mpc)
            action = numpy_action_seq[0]
            current_np_state, stable = self.eval_env.step(
                action, thresh=thresh_stable
            )
            if states is not None:
                self.eval_env._state.from_np(states[i])
                current_np_state = states[i]
                stable = i < (len(states) - 1)
            if not stable:
                if self.render:
                    np.set_printoptions(precision=3, suppress=True)
                    print("unstable")
                    # print(self.eval_env._state.as_np)
                break
            self.help_render(sleep=0)

            drone_pos = current_np_state[:3]
            drone_trajectory.append(current_np_state)

            # project to trajectory and check divergence
            drone_on_line = reference.project_on_ref(drone_pos)
            reference_trajectory.append(drone_on_line)
            div = np.linalg.norm(drone_on_line - drone_pos)
            divergences.append(div)
            if div > thresh_div:
                if self.render:
                    np.set_printoptions(precision=3, suppress=True)
                    print("state")
                    print([round(s, 2) for s in current_np_state])
                    print("trajectory:")
                    print(np.around(trajectory, 2))
                break
        if self.render:
            self.eval_env.close()
        # return trajectorie and divergences
        return (
            np.array(reference_trajectory), np.array(drone_trajectory),
            divergences
        )

    def compute_speed(self, drone_traj):
        dist = 0
        for j in range(len(drone_traj) - 1):
            dist += np.linalg.norm(drone_traj[j, :3] - drone_traj[j + 1, :3])

        time_passed = len(drone_traj) * self.dt
        speed = dist / time_passed
        return speed

    def sample_circle(self):
        possible_planes = [[0, 1], [0, 2], [1, 2]]
        plane = possible_planes[np.random.randint(0, 3, 1)[0]]
        radius = np.random.rand() * 3 + 2
        direct = np.random.choice([-1, 1])
        circle_args = {"plane": plane, "radius": radius, "direction": direct}
        return circle_args

    def eval_ref(
        self,
        reference: str,
        nr_test: int = 10,
        max_steps: int = 200,
        thresh_div=1,
        thresh_stable=1
    ):
        """
        Function to evaluate a trajectory multiple times
        """
        if nr_test == 0:
            return 0, 0
        div, stable = [], []
        for _ in range(nr_test):
            circle_args = self.sample_circle()
            _, drone_traj, divergences = self.follow_trajectory(
                reference,
                max_nr_steps=max_steps,
                thresh_div=thresh_div,
                thresh_stable=thresh_stable,
                **circle_args
            )
            div.append(np.mean(divergences))
            stable.append(len(drone_traj))

        # Output results
        print(
            "%s: Average divergence: %3.2f (%3.2f)" %
            (reference, np.mean(div), np.std(div))
        )
        print(
            "%s: Steps until divergence: %3.2f (%3.2f)" %
            (reference, np.mean(stable), np.std(stable))
        )
        return np.mean(stable), np.std(stable)

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


def load_model(model_path, epoch="", name="model_quad"):
    """
    Load model and corresponding parameters
    """
    # load std or other parameters from json
    with open(os.path.join(model_path, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, name + epoch))
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
        default="current_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-r", "--ref", type=str, default="circle", help="which trajectory"
    )
    parser.add_argument(
        '-p',
        '--points',
        type=str,
        default=None,
        help="use predefined reference"
    )
    parser.add_argument(
        "-u", "--unity", action='store_true', help="unity rendering"
    )
    parser.add_argument(
        "-f", "--flightmare", action='store_true', help="Flightmare"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    args = parser.parse_args()

    # rendering
    render = 1
    if args.unity:
        render = 0

    # load model
    model_name = args.model
    model_path = os.path.join("trained_models", "drone", model_name)
    net, param_dict = load_model(model_path, epoch=args.epoch)

    # optinally change drone speed
    # param_dict["max_drone_dist"] = .6
    # define evaluation environment
    dataset = DroneDataset(1, 1, **param_dict)
    evaluator = QuadEvaluator(
        net,
        dataset,
        render=render,
        take_every_x=5000,
        # optimizer=optim.SGD(net.parameters(), lr=0.000001, momentum=0.9),
        **param_dict
    )

    # FLIGHTMARE
    if args.flightmare:
        evaluator.eval_env = FlightmareWrapper(param_dict["dt"], args.unity)

    # Specify arguments for the trajectory
    fixed_axis = 1
    traj_args = {
        "plane": [0, 2],
        "radius": 2,
        "direction": 1,
        "thresh_div": 3,
        "thresh_stable": 1
    }
    if args.points is not None:
        from neural_control.utils.predefined_trajectories import collected_trajectories
        traj_args["points_to_traverse"] = collected_trajectories[args.points]

    # RUN
    if args.unity:
        evaluator.eval_env.env.connectUnity()

    reference_traj, drone_traj, _ = evaluator.follow_trajectory(
        args.ref, max_nr_steps=500, **traj_args
    )

    if args.unity:
        evaluator.eval_env.env.disconnectUnity()

    # EVAL
    print("Speed:", evaluator.compute_speed(drone_traj[100:300, :3]))
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
