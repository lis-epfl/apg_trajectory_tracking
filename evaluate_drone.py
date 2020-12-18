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

    def stabilize(self, nr_iters=1, render=False, max_time=300):
        collect_data = []
        actions = []
        failure_list = list()  # collect the reason for failure
        collect_runs = list()
        with torch.no_grad():
            eval_env = QuadRotorEnvBase()
            for _ in range(nr_iters):
                time_stable = 0
                eval_env.reset()
                # zero_state = np.zeros(20)
                # zero_state[9:13] = 500
                # zero_state[2] = 2
                # eval_env._state.from_np(zero_state)
                current_np_state = eval_env._state.as_np
                stable = True
                while stable and time_stable < max_time:
                    numpy_action_seq = self.predict_actions(current_np_state)
                    for nr_action in range(ROLL_OUT):
                        action = numpy_action_seq[nr_action]
                        actions.append(action)
                        # if render:
                        #     # print(np.around(current_np_state[3:6], 2))
                        #     print("action:", np.around(suggested_action, 2))
                        current_np_state, stable = eval_env.step(action)
                        collect_data.append(current_np_state)
                        if not stable:
                            # print("FAILED")
                            # print(current_torch_state)
                            # print("action", action)
                            # print(current_np_state)
                            att_stable, pos_stable = eval_env.get_is_stable(
                                current_np_state
                            )
                            # if att_stable = 1, pos_stable must be 0
                            failure_list.append(att_stable)
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
        if len(failure_list) == 0:
            failure_list = [0]
        act = np.array(actions)
        collect_data = np.array(collect_data)
        print(
            "Position was responsible in ", round(np.mean(failure_list), 2),
            "cases"
        )
        print("avg and std action", np.mean(act, axis=0), np.std(act, axis=0))
        return (
            np.mean(collect_runs), np.std(collect_runs), np.mean(failure_list),
            collect_data
        )

    def follow_trajectory(self, knots, render=False, target_change_theta=.1):
        distance_between_knots = np.linalg.norm(knots[1] - knots[0])
        torch.set_grad_enabled(False)
        # with torch.no_grad()
        eval_env = QuadRotorEnvBase()
        eval_env.reset()
        eval_env._state.set_position(knots[0])
        current_np_state = eval_env._state.as_np
        print("current state", current_np_state[:3])
        print("next state", knots[1])
        # as input to network, set next target to 0,0,0 and compute divergence
        # from current state
        target_ind = 1

        time_stable = 0
        stable = True
        while stable:
            diff_to_target = current_np_state.copy()
            diff_to_target[:3] = diff_to_target[:3] - knots[target_ind]
            # TODO: only necessary for this model where 000 is actually 002
            diff_to_target[2] += 2
            print("diff_to_target", [round(s, 2) for s in diff_to_target[:3]])

            numpy_action_seq = self.predict_actions(diff_to_target)
            for nr_action in range(ROLL_OUT):
                # retrieve next action
                action = numpy_action_seq[nr_action]
                # take step in environment
                current_np_state, stable = eval_env.step(action)
                # att_stable, pos_stable = eval_env.get_is_stable(
                #     current_np_state
                # )
                if render:
                    eval_env.render()
                    time.sleep(.1)
                time_stable += 1
                # if time_stable % 50 == 0:
                #     print([round(s, 2) for s in current_np_state[:3]])
                if not stable:
                    print("FAILED")
                    break
            # update diff to target: Find out whether we overtook the target
            current_pos = current_np_state[:3]
            # if the drone has passed the current target, the scalar product is
            # smaller zero
            if target_ind == len(knots) - 1:
                print("reached end")
                continue
            # scalar_product = np.dot(
            #     knots[target_ind + 1] - current_pos,
            #     knots[target_ind] - current_pos
            # )
            # # TODO: normalize and check whether scalar product is <45
            # print("scalar product", scalar_product)
            # if scalar_product <= 0.5:
            print()
            print("pos", [round(s, 2) for s in current_pos])
            print("left", np.linalg.norm(knots[target_ind] - current_pos))
            if np.linalg.norm(
                knots[target_ind] - current_pos
            ) < distance_between_knots * target_change_theta:
                # aim at next target
                target_ind += 1
                print("--------- go to next target:", target_ind, "------")
                time.sleep(1)

    @staticmethod
    def random_trajectory(distance, number_knots):
        start = (np.random.rand(3) - .5) * distance  #).astype(int)
        end = (np.random.rand(3) - .5) * distance  #).astype(int)
        start_to_end_unit = (end - start) / (number_knots - 1)
        knots = np.array(
            [start + i * start_to_end_unit for i in range(number_knots)]
        )
        # make height to be at least 1
        knots[:, 2] += max(1 - np.min(knots[:, 2]), 0)
        return knots


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
    knots = QuadEvaluator.random_trajectory(2, 10)
    knots[:, :2] *= .5
    print("start, end")
    print(knots[0], knots[-1])
    evaluator.follow_trajectory(knots, render=True)
