import os
import time
import argparse
import json
import numpy as np
import torch

from environments.drone_env import QuadRotorEnvBase
from utils.plotting import plot_state_variables
from dataset import raw_states_to_torch
from models.resnet_like_model import Net
from drone_loss import drone_loss_function

ROLL_OUT = 1
ACTION_DIM = 4


class QuadEvaluator():

    def __init__(self, model, mean=0, std=1):
        self.mean = mean
        # self.mean[2] -= 2  # for old models
        self.std = std
        self.net = model

    def predict_actions(self, current_np_state):
        current_torch_state = raw_states_to_torch(
            current_np_state, normalize=True, mean=self.mean, std=self.std
        )
        suggested_action = self.net(current_torch_state)
        suggested_action = torch.sigmoid(suggested_action)[0]

        suggested_action = torch.reshape(suggested_action, (-1, ACTION_DIM))
        numpy_action_seq = suggested_action.numpy()
        return numpy_action_seq

    def stabilize(
        self, nr_iters=1, render=False, max_time=300, start_loss=100
    ):
        collect_data = []
        pos_loss_list = list()  # collect the reason for failure
        collect_runs = list()
        with torch.no_grad():
            eval_env = QuadRotorEnvBase()
            for _ in range(nr_iters):
                time_stable = 0
                eval_env.render_reset()
                # zero_state = np.zeros(20)
                # zero_state[9:13] = 500
                # zero_state[2] = 2
                # eval_env._state.from_np(zero_state)
                stable = True
                while stable and time_stable < max_time:
                    current_np_state = eval_env._state.as_np.copy()
                    current_np_state[2] -= 2  # correct for height
                    if time_stable > start_loss:
                        pos_loss_list.append(np.absolute(current_np_state[:3]))
                    numpy_action_seq = self.predict_actions(current_np_state)
                    for nr_action in range(ROLL_OUT):
                        action = numpy_action_seq[nr_action]
                        # if render:
                        #     # print(np.around(current_np_state[3:6], 2))
                        #     print("action:", np.around(suggested_action, 2))
                        current_np_state, stable = eval_env.step(action)
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
                        eval_env.render()
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

    def follow_trajectory(
        self,
        knots,
        data_list,
        render=False,
        target_change_theta=.5,
        max_iters=300
    ):
        """
        Evaluate the ability of the drone to follow a trajectory defined by
        knots
        """
        main_target = knots[-1]
        distance_between_knots = np.linalg.norm(knots[1] - knots[0])

        # Set up environment
        eval_env = QuadRotorEnvBase()
        eval_env.zero_reset(*tuple(knots[0]))
        current_np_state = eval_env._state.as_np

        target_ind = 1
        time_stable = 0
        stable = True
        min_distance_to_target = np.inf
        while stable and time_stable < max_iters:
            # as input to network, consider target as 0,0,0 - input difference
            diff_to_target = current_np_state.copy()
            diff_to_target[:3] = diff_to_target[:3] - knots[target_ind]

            # Predict actions and execute
            data_list.append(diff_to_target)
            numpy_action_seq = self.predict_actions(diff_to_target)
            for nr_action in range(ROLL_OUT):
                # retrieve next action
                action = numpy_action_seq[nr_action]
                # take step in environment
                current_np_state, stable = eval_env.step(action)

                # output
                if render:
                    eval_env.render()
                    time.sleep(.1)
                    if time_stable % 30 == 0:
                        print()
                        current_pos = current_np_state[:3]
                        print("pos", [round(s, 2) for s in current_pos])
                        print(
                            "left",
                            np.linalg.norm(knots[target_ind] - current_pos)
                        )
                        print(
                            "diff_to_target",
                            [round(s, 2) for s in diff_to_target[:3]]
                        )
                time_stable += 1
                if not stable:
                    # print("FAILED")
                    break

            # Log the difference to the target
            current_pos = current_np_state[:3]
            diff_to_main = np.sqrt(np.sum((main_target - current_pos)**2))
            if diff_to_main < min_distance_to_target:
                min_distance_to_target = diff_to_main

            if target_ind == len(knots) - 1:  # no need to proceed to next t
                continue

            # Check if the drone has passed the current target or is close
            use_next_target = np.linalg.norm(
                knots[target_ind] - current_pos
            ) < distance_between_knots * target_change_theta
            if use_next_target:
                # aim at next target
                target_ind += 1
                # print("--------- go to next target:", target_ind, "------")
                time.sleep(1)
        return min_distance_to_target, time_stable, data_list

    def evaluate(self, nr_hover_iters=5, nr_traj_iters=10):
        with torch.no_grad():
            data_list = []

            # hover:
            progress_list, timesteps_list = list(), list()
            for _ in range(nr_hover_iters):
                hover_knots = QuadEvaluator.hover_trajectory()
                target_dist = np.linalg.norm(hover_knots[-1] - hover_knots[0])
                progress, timesteps, data_list = self.follow_trajectory(
                    hover_knots, data_list
                )
                progress_list.append(1 - progress / target_dist)
                timesteps_list.append(timesteps)
            print(
                "Hover normalized avg progress to target: {:.2f} ({:.2f})\
                    - Hover timesteps before fail: {:.2f} ({:.2f})".format(
                    np.mean(progress_list), np.std(progress_list),
                    np.mean(timesteps_list), np.std(timesteps_list)
                )
            )

            # follow trajectory
            progress_list, timesteps_list = list(), list()
            for _ in range(nr_traj_iters):
                traj_knots = QuadEvaluator.random_trajectory(1.5)
                overall_distance = np.linalg.norm(
                    traj_knots[-1] - traj_knots[0]
                )
                missing, timesteps, data_list = self.follow_trajectory(
                    traj_knots, data_list
                )
                progress_list.append(1 - missing / overall_distance)
                timesteps_list.append(timesteps)
            print(
                "Trajectory normalized avg progress to target: {:.2f} ({:.2f})\
                    - Trajectory timesteps before fail: {:.2f} ({:.2f})".
                format(
                    np.mean(progress_list), np.std(progress_list),
                    np.mean(timesteps_list), np.std(timesteps_list)
                )
            )
            data_list = np.array(data_list)
        return np.mean(progress_list), np.std(progress_list), data_list

    @staticmethod
    def random_trajectory(step_size, diff=5):
        # if diff = 5, then x,y,z are between 0 and 5
        start = np.random.rand(3) * diff
        end = np.random.rand(3) * diff
        dist = np.sqrt(np.sum((end - start)**2))
        # number of knots required to have step_size distance inbetween
        number_knots = int(np.ceil(dist / step_size)) + 1

        start_to_end_unit = (end - start) / (number_knots - 1)
        knots = np.array(
            [start + i * start_to_end_unit for i in range(number_knots)]
        )
        # make height to be at least 1
        knots[:, 2] += max(1 - np.min(knots[:, 2]), 0)
        return knots

    @staticmethod
    def hover_trajectory():
        eval_env = QuadRotorEnvBase()
        eval_env.render_reset()
        start = eval_env._state.as_np[:3]
        end = [0, 0, 2]
        return np.array([start, end])


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
    net.eval()

    evaluator = QuadEvaluator(
        net,
        mean=np.array(param_dict["mean"]),
        std=np.array(param_dict["std"])
    )
    # # watch
    # _, _, _, collect_data = evaluator.stabilize(nr_iters=1, render=True)
    # # compute stats
    # success_mean, success_std, _, _ = evaluator.stabilize(
    #     nr_iters=100, render=False
    # )
    # print(success_mean, success_std)
    # # plot_state_variables(
    # #     collect_data, save_path=os.path.join(model_path, "evaluation.png")
    # # )

    # test trajectory
    knots = QuadEvaluator.random_trajectory(1.5)
    # hover_trajectory()
    # random_trajectory(10, 4)
    print("Knots:")
    print(np.around(knots, 2))
    with torch.no_grad():
        evaluator.follow_trajectory(knots, [], render=True)

    evaluator.evaluate()
