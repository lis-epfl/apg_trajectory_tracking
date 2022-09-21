import json
import os
from xml.etree.ElementInclude import default_loader
import numpy as np
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.environments.rl_envs import (
    CartPoleEnvRL, WingEnvRL, QuadEnvRL
)
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.models.hutter_model import Net
from neural_control.models.simple_model import Net as NetPole
from neural_control.plotting import plot_wing_pos_3d
from neural_control.trajectory.q_funcs import project_to_line

# PARAMS
fixed_wing_dt = 0.05
cartpole_dt = 0.05
quad_dt = 0.1
quad_speed = 0.2
quad_horizon = 10
curriculum = True


class EvalCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps
    :param save_freq: (int)
    :param save_path: (str) Path to the folder where the model will be saved.
    :param name_prefix: (str) Common prefix to the saved models
    """

    def __init__(
        self,
        eval_func,  # function to evaluate model
        eval_env,
        eval_freq: int,
        save_path: str,
        nr_iters=10,
        # eval_key="mean_div",
        # eval_up_down=-1,
        verbose=0
    ):
        super(EvalCallback, self).__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_func = eval_func
        self.eval_env = eval_env
        self.save_path = save_path
        self.nr_iters = nr_iters

        # self.best_perf = 0 if eval_up_down == 1 else np.inf
        self.res_dict = defaultdict(list)

    def _on_step(self) -> bool:
        if (self.n_calls - 1) % self.eval_freq == 0:
            if curriculum and self.eval_env.thresh_div < 3:
                self.eval_env.thresh_div += .05
                print("increased thresh div", self.eval_env.thresh_div)
            # evaluate
            _, res_step = self.eval_func(
                self.model, self.eval_env, nr_iters=self.nr_iters
            )
            for key in res_step.keys():
                self.res_dict[key].append(res_step[key])
            self.res_dict["samples"].append(self.num_timesteps)

            # save every time (TODO: change to saving best?)
            path = self.save_path + '_{}_steps'.format(self.num_timesteps)
            self.model.save(path)
            print("model saved at ", path)
            if self.verbose > 1:
                print("Saving model checkpoint to {}".format(path))
        return True


def train_main(
    model_path,
    env,
    evaluate_func,
    load_model=None,
    total_timesteps=50000,
    eval_freq=10000
):
    # make directory
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    save_name = os.path.join(model_path, "rl")

    if load_model is None:
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log="./rl_quad_tensorboard/"
        )
    else:
        model = PPO.load(load_model, env=env)

    eval_callback = EvalCallback(
        evaluate_func,
        env,
        eval_freq=eval_freq,
        save_path=save_name,
        nr_iters=40
    )
    try:
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    except KeyboardInterrupt:
        pass
    with open(save_name + "_res.json", "w") as outfile:
        json.dump(eval_callback.res_dict, outfile)
    model.save(save_name + "_final")


# ------------------ CartPole -----------------------


def train_cartpole(model_path, load_model=None, modified_params={}):
    dyn = CartpoleDynamics(modified_params=modified_params)
    env = CartPoleEnvRL(dyn, dt=cartpole_dt)
    train_main(
        model_path,
        env,
        evaluate_cartpole,
        load_model=load_model,
        total_timesteps=500000,
        eval_freq=10000
    )


def evaluate_cartpole(model, env, max_steps=250, nr_iters=1, render=0):
    states, actions = [], []
    num_stable = []
    for j in range(nr_iters):
        obs = env.reset()
        for i in range(max_steps):
            if isinstance(model, NetPole):  # DP
                with torch.no_grad():
                    obs_torch = torch.from_numpy(np.expand_dims(obs,
                                                                0)).float()
                    suggested_action = model(obs_torch)
                    suggested_action = torch.reshape(suggested_action, (10, 1))
                    action = suggested_action[0].numpy()
            else:
                action, _states = model.predict(obs, deterministic=True)
            actions.append(action)
            obs, rewards, done, info = env.step(action)
            states.append(env.state)
            if render:
                env.render()
            if done:
                break

        num_stable.append(i)
    states = np.array(states)
    actions = np.array(actions)
    mean_vel = np.mean(np.absolute(states[:, 1]))
    std_vel = np.std(np.absolute(states[:, 1]))
    mean_stable = np.mean(num_stable)
    std_stable = np.std(num_stable)
    print("Average stable: %3.2f (%3.2f)" % (mean_stable, std_stable))
    print("Average velocity: %3.2f (%3.2f)" % (mean_vel, std_vel))
    res_step = {
        "mean_vel": mean_vel,
        "std_vel": std_vel,
        "mean_stable": mean_stable,
        "std_stable": std_stable
    }
    # plt.hist(actions)
    # plt.show()
    return 0, res_step


def test_rl_cartpole(save_name, modified_params={}, max_steps=250):
    dyn = CartpoleDynamics(modified_params=modified_params)
    env = CartPoleEnvRL(dyn, dt=cartpole_dt)
    model = PPO.load(save_name)
    evaluate_cartpole(model, env, max_steps, nr_iters=30, render=0)


def test_ours_cartpole(model_path, modified_params={}, max_steps=500):
    model = torch.load(os.path.join(model_path, "model_cartpole"))
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)
    dyn = CartpoleDynamics(modified_params=modified_params)
    # param_dict["speed_factor"] = .2
    env = CartPoleEnvRL(dyn, **param_dict)
    evaluate_cartpole(model, env, max_steps, nr_iters=40, render=0)


# ------------------ Fixed wing drone -----------------------


def evaluate_wing(model=None, env=None, max_steps=1000, nr_iters=1, render=0):
    # TODO: merge evaluate functions
    if env is None:
        dyn = FixedWingDynamics()
        env = WingEnvRL(dyn, dt=0.05)

    div_target = []
    np.set_printoptions(precision=3, suppress=1)
    for j in range(nr_iters):
        obs = env.reset(x_dist=50, x_std=5)
        if render:
            print(f"iter {j}:", env.target_point)
        trajectory = []
        for i in range(max_steps):
            if model is not None:
                # OURS
                if isinstance(model, Net):
                    obs_state, obs_ref = env.prepare_obs()
                    with torch.no_grad():
                        suggested_action = model(obs_state, obs_ref)
                        suggested_action = torch.sigmoid(suggested_action)[0]
                        suggested_action = torch.reshape(
                            suggested_action, (10, 4)
                        )
                        action = suggested_action[0].numpy()
                else:
                    # RL
                    action, _states = model.predict(obs, deterministic=True)
                # print(action)
            else:
                action_prior = np.array([.25, .5, .5, .5])
                sampled_action = np.random.normal(scale=.15, size=4)
                action = np.clip(sampled_action + action_prior, 0, 1)

            obs, rewards, done, info = env.step(action)
            # print(env.state[:3], env.get_divergence())
            # print()
            trajectory.append(env.state)

            if render:
                env.render()
            if done:
                if env.state[0] < 20:
                    div_target.append(
                        np.linalg.norm(env.state[:3] - env.target_point)
                    )
                else:
                    target_on_traj = project_to_line(
                        trajectory[-2][:3], env.state[:3], env.target_point
                    )
                    div_target.append(
                        np.linalg.norm(target_on_traj - env.target_point)
                    )
                if render:
                    print("last state", env.state[:3], "div", div_target[-1])
                break

    print(
        "Average error: %3.2f (%3.2f)" %
        (np.mean(div_target), np.std(div_target))
    )
    return np.array(trajectory), {
        "mean_div": np.mean(div_target),
        "std_div": np.std(div_target)
    }


def train_wing(model_path, load_model=None, modified_params={}):
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, fixed_wing_dt)

    eval_freq = 10000 if load_model is None else 200
    train_main(
        model_path,
        env,
        evaluate_wing,
        load_model=load_model,
        total_timesteps=500000,
        eval_freq=eval_freq
    )


def test_rl_wing(save_name, modified_params={}, max_steps=1000, nr_iters=50):
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, fixed_wing_dt)
    model = PPO.load(save_name)
    if nr_iters > 1:
        trajectory, _ = evaluate_wing(
            model, env, max_steps, nr_iters=nr_iters, render=0
        )
    else:
        trajectory, _ = evaluate_wing(
            model, env, max_steps, nr_iters=1, render=1
        )

    return trajectory


def test_ours_wing(model_path, modified_params={}, max_steps=1000):
    model = torch.load(os.path.join(model_path, "model_wing"))
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = WingEnvRL(dyn, **param_dict)
    evaluate_wing(model, env, max_steps, nr_iters=40, render=0)


# ------------------ Quadrotor -----------------------


def evaluate_quad(model, env, max_steps=500, nr_iters=1, render=0):
    divergences = []
    num_steps = []
    np.set_printoptions(precision=3, suppress=1)
    for j in range(nr_iters):
        obs = env.reset()
        drone_trajectory, avg_div = [], []
        for i in range(max_steps):
            if isinstance(model, Net):  # DP
                obs_state, obs_ref = env.prepare_obs()
                with torch.no_grad():
                    suggested_action = model(obs_state, obs_ref)
                    suggested_action = (
                        torch.sigmoid(suggested_action)[0] * 2 - 1
                    )
                    suggested_action = torch.reshape(suggested_action, (10, 4))
                    action = suggested_action[0].numpy()
            else:  # RL
                action, _states = model.predict(obs, deterministic=True)

            obs, rewards, done, info = env.step(action)
            if render:
                env.render()
            avg_div.append(env.get_divergence())
            drone_trajectory.append(env.state)
            if done:
                break
        num_steps.append(len(drone_trajectory))
        divergences.append(np.mean(avg_div))

    full_runs = np.array(num_steps) == 489
    if np.sum(full_runs) > 0:
        div_full_runs = np.array(divergences)[full_runs]
        print(
            "Error full runs: %3.2f (%3.2f)" %
            (np.mean(div_full_runs), np.std(div_full_runs))
        )
    print(
        "Tracking error: %3.2f (%3.2f)" %
        (np.mean(divergences), np.std(divergences))
    )
    print(
        "Number steps: %3.2f (%3.2f)" %
        (np.mean(num_steps), np.std(num_steps))
    )
    res_step = {
        "mean_div": np.mean(divergences),
        "std_div": np.std(divergences),
        "mean_steps": np.mean(num_steps),
        "std_steps": np.std(num_steps)
    }
    return drone_trajectory, res_step


def test_rl_quad(save_name, modified_params={}, max_steps=1000, nr_iters=30):
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(
        dyn, quad_dt, speed_factor=quad_speed, nr_actions=quad_horizon
    )
    env.thresh_div = 3
    model = PPO.load(save_name)
    if nr_iters == 1:
        _ = evaluate_quad(model, env, max_steps, nr_iters=1, render=1)
    else:
        _ = evaluate_quad(model, env, max_steps, nr_iters=nr_iters, render=0)


def test_ours_quad(model_path, modified_params={}, max_steps=500):
    model = torch.load(os.path.join(model_path, "model_quad"))
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as outfile:
        param_dict = json.load(outfile)
    dyn = FlightmareDynamics(modified_params=modified_params)
    # param_dict["speed_factor"] = .2
    env = QuadEnvRL(dyn, **param_dict)
    evaluate_quad(model, env, max_steps, nr_iters=40, render=0)


def train_quad(model_path, load_model=None, modified_params={}):
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = QuadEnvRL(
        dyn, quad_dt, speed_factor=quad_speed, nr_actions=quad_horizon
    )
    train_main(
        model_path,
        env,
        evaluate_quad,
        load_model=load_model,
        total_timesteps=2000000,
        eval_freq=10000
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--system",
        default="quadrotor",
        help="cartpole, fixed_wing or quadrotor"
    )
    parser.add_argument(
        "-a",
        "--eval_iters",
        default=1,
        help="nr of iterations for evaluation"
    )
    args = parser.parse_args()
    # ------------------ CartPole -----------------------
    if args.system == "cartpole":
        pass
    # ------------------ Fixed wing drone -----------------------
    elif args.system == "fixed_wing":
        # Final: BL evaluation PPO
        load_name = "trained_models/wing/ppo_bl/rl_final"
        scenario = {}
        # train_wing(save_name, load_model=load_name, modified_params=scenario)
        test_rl_wing(
            load_name, modified_params=scenario, nr_iters=args.eval_iters
        )
    elif args.system == "quadrotor":
        # ------------------ Quadrotor -----------------------
        # # Final: BL evaluation PPO:
        load_name = "trained_models/quad/ppo/rl_final"
        test_rl_quad(load_name, modified_params={}, nr_iters=args.eval_iters)
    else:
        raise NotImplementedError(
            "System must be one of cartpole, fixed_wing or quadrotor"
        )
