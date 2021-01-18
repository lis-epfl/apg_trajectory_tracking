import os
import time
import argparse
import json
import numpy as np
import torch
import pickle

from environments.drone_env import QuadRotorEnvBase
from utils.plotting import plot_state_variables, plot_trajectory
from utils.trajectory import sample_points_on_straight, sample_to_input, np_project_line, eval_get_reference
from dataset import raw_states_to_torch
# from models.resnet_like_model import Net
from drone_loss import drone_loss_function

ROLL_OUT = 1
ACTION_DIM = 4

# Use cuda if available
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadEvaluator():

    def __init__(self, model, mean=0, std=1, horizon=5, max_drone_dist=0.1):
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

    def stabilize(
        self, nr_iters=1, render=False, max_time=300, start_loss=200
    ):
        """
        Measure nr_iters times how long the drone can hover without falling.
        Arguments:
            nr_iters (int): Number of runs (multiple to compute statistics)
            render (bool): if true, showing a simple drone simulation
            max_time (int): Maximum number of steps to hover
            start_loss (int): At this point we start to record the average
                divergence from the target for evaluation --> In the optimal
                case the drone has stabilized around the target at this time
        Returns:
            Average number of steps before failure
            Standard deviation of steps before failure
            collect_data: Array with all encountered states
        """
        collect_data = []
        pos_loss_list = list()  # collect the reason for failure
        collect_runs = list()
        with torch.no_grad():
            for _ in range(nr_iters):
                time_stable = 0
                # Reset and run until failing or reaching max_time
                self.eval_env.render_reset()
                stable = True
                while stable and time_stable < max_time:
                    current_np_state = self.eval_env._state.as_np.copy()
                    current_np_state[2] -= 2  # correct for height
                    if time_stable > start_loss:
                        pos_loss_list.append(np.absolute(current_np_state[:3]))
                    numpy_action_seq = self.predict_actions(current_np_state)
                    for nr_action in range(ROLL_OUT):
                        action = numpy_action_seq[nr_action]
                        # if render:
                        #     # print(np.around(current_np_state[3:6], 2))
                        #     print("action:", np.around(suggested_action, 2))
                        current_np_state, stable = self.eval_env.step(action)
                        if time_stable > 20:
                            collect_data.append(current_np_state)
                        if not stable:
                            break
                        # if render:
                        #     print(current_np_state[:3])
                        # count nr of actions that the drone can hover
                        time_stable += 1
                    if render:
                        # .numpy()[0]
                        print([round(s, 2) for s in current_np_state])
                        self.eval_env.render()
                        time.sleep(.1)
                collect_runs.append(time_stable)
        collect_data = np.array(collect_data)
        if len(pos_loss_list) > 0:
            print(
                "average deviation from target pos:", [
                    round(s, 2)
                    for s in np.mean(np.array(pos_loss_list), axis=0)
                ]
            )
        return (np.mean(collect_runs), np.std(collect_runs), 0, collect_data)

    def eval_traj_input(
        self,
        threshold_divergence,
        nr_test_data=5,
        max_nr_steps=200,
        step_size=0.01,
        render=False
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
                    trajectory = eval_get_reference(
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
                    if render:
                        print(current_np_state[:3], trajectory[0, :3])
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
                    if render:
                        # print([round(s, 2) for s in current_np_state])
                        self.eval_env._state.set_position(
                            current_np_state[:3] + np.array([0, 0, 1])
                        )
                        self.eval_env.render()
                        self.eval_env._state.set_position(current_np_state[:3])
                        time.sleep(.2)

                    drone_on_line = np_project_line(
                        a_on_line, b_on_line, current_np_state[:3]
                    )
                    initial_trajectory.append(drone_on_line)
                    div = np.linalg.norm(drone_on_line - current_np_state[:3])
                    if div > threshold_divergence:
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

    evaluator = QuadEvaluator(
        net,
        mean=np.array(param_dict["mean"]),
        std=np.array(param_dict["std"]),
        horizon=param_dict["horizon"],
        max_drone_dist=param_dict["max_drone_dist"]
    )

    threshold_divergence = 5 * param_dict["max_drone_dist"]
    # Straight with reference as input
    try:
        initial_trajectory, drone_trajectory = evaluator.eval_traj_input(
            threshold_divergence,
            nr_test_data=1,
            render=True,
            max_nr_steps=300,
            step_size=param_dict["step_size"],
        )
        plot_trajectory(
            initial_trajectory, drone_trajectory,
            os.path.join(model_path, "traj.png")
        )
        # evaluator.eval_traj_input(threshold_divergence, nr_test_data=50,
        # render=False, max_nr_steps=200, step_size=param_dict["step_size"])
    except KeyboardInterrupt:
        evaluator.eval_env.close()
    # evaluator.evaluate()
