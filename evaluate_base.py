import os
import json
import numpy as np
import torch

from neural_control.plotting import plot_success
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics


def load_model_params(model_path, name="model_quad", epoch=""):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print("Load old config..")
        config_path = os.path.join(model_path, "param_dict.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)

    net = torch.load(os.path.join(model_path, name + epoch))
    net.eval()
    return net, param_dict


global last_actions
last_actions = np.zeros((10, 4))


def average_action(action, step, do_avg_act=False):
    global last_actions
    if do_avg_act:
        # make average action
        if step == 0:
            last_actions = action.copy()
        else:
            last_actions = np.roll(last_actions, -1, axis=0)
            # rolling mean
            weight = np.ones((10, 1))
            # np.expand_dims(9 - np.arange(10), 1)
            # np.ones((10, 1)) # for giving more weight to newer states

            last_actions = (last_actions * weight + action) / (weight + 1)
        # print("actions", action)
        # print("last actions", last_actions)\
        use_action = last_actions[0]
    else:
        use_action = action[0]
    return use_action


def increase_param(default_val, inc):
    # first case: param is array
    if isinstance(default_val, list):
        new_val = (np.array(default_val) * inc).astype(float)
        # if all zero, add instead
        if not np.any(new_val):
            new_val += (inc - 1)
    else:
        new_val = float(default_val * inc)
        if new_val == 0:
            new_val += (inc - 1)
    return new_val


def run_mpc_analysis(
    evaluator, system="fixed_wing", out_path="../presentations/analysis"
):
    """
    Run eval function with mpc multiple times and plot the results
    Args:
        evaluator (Evaluator): fully initialized environment with controller
    """
    with open(f"neural_control/dynamics/config_{system}.json", "r") as inf:
        parameters = json.load(inf)

    increase_factors = np.arange(1, 2, .1)
    for key, default_val in parameters.items():
        # for key in ["mass"]:
        #     default_val = parameters[key]

        if key == "g" or key == "gravity":
            # gravity won't change ;)
            continue
        default_val = parameters[key]

        print(
            f"\n-------------{key} (with default {default_val}) ------------"
        )
        mean_list, std_list = [], []
        for inc in increase_factors:
            new_val = increase_param(default_val, inc)

            modified_params = {key: new_val}
            print("\n ", round(inc, 2), "modified:", modified_params)

            if system == "fixed_wing":
                evaluator.eval_env.dynamics = FixedWingDynamics(
                    modified_params=modified_params
                )
            elif system == "quad":
                evaluator.eval_env.dynamics = FlightmareDynamics(
                    modified_params=modified_params
                )

            mean_dist, std_dist = evaluator.run_eval(nr_test=20)
            mean_list.append(mean_dist)
            std_list.append(std_dist)
        x = np.array(increase_factors)
        plot_success(
            x, mean_list, std_list, os.path.join(out_path, key + "_mpc.jpg")
        )
