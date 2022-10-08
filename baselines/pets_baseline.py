#!/usr/bin/env python
# coding: utf-8

# # THIS CODE IS TAKEN FROM THE MBRL LIBRARY
# See https://github.com/facebookresearch/mbrl-lib (MIT license)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import omegaconf
import time
import json
import os

import mbrl.env.reward_fns as reward_fns
import mbrl.env.termination_fns as termination_fns
import mbrl.models as models
import mbrl.planning as planning
import mbrl.util.common as common_util
import mbrl.util as util

SYSTEM = "fixed_wing"

modified_params = {}
if SYSTEM == "cartpole":
    import neural_control.environments.rl_envs as cartpole_env
    from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
    dyn = CartpoleDynamics(modified_params=modified_params)
    env = cartpole_env.CartPoleEnvRL(dyn)
    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.cartpole
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.cartpole
elif SYSTEM == "quad":
    import neural_control.environments.rl_envs as quad_env
    from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
    quad_dt = 0.1
    quad_speed = 0.2
    dyn = FlightmareDynamics(modified_params=modified_params)
    env = quad_env.QuadEnvRL(dyn, quad_dt, speed_factor=quad_speed)
    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.quad
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.quad
elif SYSTEM == "fixed_wing":
    import neural_control.environments.rl_envs as wing_env
    from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
    dyn = FixedWingDynamics(modified_params=modified_params)
    env = wing_env.WingEnvRL(dyn, 0.05, div_in_obs=True, thresh_div=.5)
    # This functions allows the model to evaluate the true rewards given an observation
    reward_fn = reward_fns.fixed_wing
    # This function allows the model to know if an observation should make the episode end
    term_fn = termination_fns.fixed_wing

# TODO!
# This example uses a probabilistic ensemble. You can also use a fully deterministic model with class GaussianMLP by setting ensemble_size=1, and deterministic=True.

# SPECIFY
model_load_path = None  # "trained_models/out_mbrl/fixed_wing_smallthresh_2/eps_64_4724/"
model_save_path = "trained_models/out_mbrl/test_halfcheetah"
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
trial_length = 200  # 200
num_trials = 200  # 10
ensemble_size = 5

mpl.rcParams.update({"font.size": 16})

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# # Creating the environment
#
# First we instantiate the environment and specify which reward function and termination function to use with the gym-like environment wrapper, along with some utility objects. The termination function tells the wrapper if an observation should cause an episode to end or not, and it is an input used in some algorithms, like [MBPO](https://github.com/JannerM/mbpo/blob/master/mbpo/static/halfcheetah.py). The reward function is used to compute the value of the reward given an observation, and it's used by some algorithms, like [PETS](https://github.com/kchua/handful-of-trials/blob/77fd8802cc30b7683f0227c90527b5414c0df34c/dmbrl/controllers/MPC.py#L65).

seed = 0
env.seed(seed)
rng = np.random.default_rng(seed=0)
generator = torch.Generator(device=device)
generator.manual_seed(seed)
obs_shape = env.observation_space.shape
act_shape = env.action_space.shape

# # Hydra configuration
#
# MBRL-Lib uses [Hydra](https://github.com/facebookresearch/hydra) to manage configurations. For the purpose of this example, you can think of the configuration object as a dictionary with key/value pairs--and equivalent attributes--that specify the model and algorithmic options. Our toolbox expects the configuration object to be organized as follows:

# count steps in environment
global step_counter
step_counter = 0

# Everything with "???" indicates an option with a missing value.
# Our utility functions will fill in these details using the
# environment information
cfg_dict = {
    # dynamics model configuration
    "dynamics_model":
        {
            "model":
                {
                    "_target_": "mbrl.models.GaussianMLP",
                    "device": device,
                    "num_layers": 3,
                    "ensemble_size": ensemble_size,
                    "hid_size": 200,
                    "use_silu": True,
                    "in_size": "???",
                    "out_size": "???",
                    "deterministic": False,
                    "propagation_method": "fixed_model"
                }
        },
    # options for training the dynamics model
    "algorithm":
        {
            "learned_rewards": False,
            "target_is_delta": True,
            "normalize": True,
        },
    # these are experiment specific options
    "overrides":
        {
            "trial_length": trial_length,
            "num_steps": num_trials * trial_length,
            "model_batch_size": 32,
            "validation_ratio": 0.05
        }
}
cfg = omegaconf.OmegaConf.create(cfg_dict)

# <div class="alert alert-block alert-info"><b>Note: </b> This example uses a probabilistic ensemble. You can also use a fully deterministic model with class GaussianMLP by setting ensemble_size=1, and deterministic=True. </div>

# # Creating a dynamics model
#
# Given the configuration above, the following two lines of code create a wrapper for 1-D transition reward models, and a gym-like environment that wraps it, which we can use for simulating the real environment. The 1-D model wrapper takes care of creating input/output data tensors to the underlying NN model (by concatenating observations, actions and rewards appropriately), normalizing the input data to the model, and other data processing tasks (e.g., converting observation targets to deltas with respect to the input observation).

# Create a 1-D dynamics model for this environment
dynamics_model = common_util.create_one_dim_tr_model(
    cfg, obs_shape, act_shape, model_dir=model_load_path
)
# load_weights_from_path TODO specifiy here
# Create a gym-like environment to encapsulate the model
model_env = models.ModelEnv(
    env, dynamics_model, term_fn, reward_fn, generator=generator
)

# # Create a replay buffer
#
# We can create a replay buffer for this environment an configuration using the following method

replay_buffer = common_util.create_replay_buffer(
    cfg, obs_shape, act_shape, rng=rng
)

# We can now populate the replay buffer with random trajectories of a desired length, using a single function call to `util.rollout_agent_trajectories`. Note that we pass an agent of type `planning.RandomAgent` to generate the actions; however, this method accepts any agent that is a subclass of `planning.Agent`, allowing changing exploration strategies with minimal changes to the code.

common_util.rollout_agent_trajectories(
    env,
    trial_length,  # initial exploration steps
    planning.RandomAgent(env),
    {},  # keyword arguments to pass to agent.act()
    replay_buffer=replay_buffer,
    trial_length=trial_length
)

print("# samples stored", replay_buffer.num_stored)

# # CEM Agent
#
# The following config object and the subsequent function call create an agent that can plan using the Cross-Entropy Method over the model environment created above. When calling `planning.create_trajectory_optim_agent_for_model`, we also specify how many particles to use when propagating model uncertainty, as well as the uncertainty propagation method, "fixed_model", which corresponds to the method TS$\infty$ in the PETS paper.

agent_cfg = omegaconf.OmegaConf.create(
    {
        # this class evaluates many trajectories and picks the best one
        "_target_": "mbrl.planning.TrajectoryOptimizerAgent",
        "planning_horizon": 15,
        "replan_freq": 1,
        "verbose": False,
        "action_lb": "???",
        "action_ub": "???",
        # this is the optimizer to generate and choose a trajectory
        "optimizer_cfg":
            {
                "_target_": "mbrl.planning.CEMOptimizer",
                "num_iterations": 5,
                "elite_ratio": 0.1,
                "population_size": 500,
                "alpha": 0.1,
                "device": device,
                "lower_bound": "???",
                "upper_bound": "???",
                "return_mean_elites": True
            }
    }
)

agent = planning.create_trajectory_optim_agent_for_model(
    model_env, agent_cfg, num_particles=20
)

# # Running PETS

# Having a model and an agent, we can now run PETS with a simple loop and a few function calls. The first code block creates a callback to pass to the model trainer to accumulate the training losses and validation scores observed. The second block is just a utility function to update the agent's visualization.

train_losses = []
val_scores = []
timestamps = []


def train_callback(
    _model, _total_calls, _epoch, tr_loss, val_score, _best_val
):
    train_losses.append(tr_loss)
    val_scores.append(
        val_score.mean().item()
    )  # this returns val score per ensemble model


# Create a trainer for the model
model_trainer = models.ModelTrainer(
    dynamics_model, optim_lr=1e-3, weight_decay=5e-5
)


# print(dynamics_model.state_dict)
# print(agent)
# Create visualization objects
# fig, axs = plt.subplots(
#     1, 2, figsize=(14, 3.75), gridspec_kw={"width_ratios": [1, 1]}
# )
# ax_text = axs[0].text(300, 50, "")
def make_dict_and_save():
    out_dict = {}
    out_dict["div_to_target"] = list(div_to_target)
    out_dict["train_losses"] = list(train_losses)
    out_dict["all_rewards"] = list(all_rewards)
    out_dict["timestamps"] = list(timestamps)
    out_dict["val_scores"] = list(val_scores)
    out_dict["all_rewards"] = list(all_rewards)
    out_dict["sum_rewards"] = list(sum_rewards)
    out_dict["episode_lens_list"] = list(episode_lens_list)
    out_dict["step_counter_list"] = list(step_counter_list)
    out_dict["velocity_list"] = list(eval_metric_list)

    with open(os.path.join(model_save_path, "out_res.json"), "w") as outfile:
        json.dump(out_dict, outfile)


# Main PETS loop
all_rewards = []
sum_rewards = []
step_counter_list = []
div_to_target = []
eval_metric_list = []
episode_lens_list = []
step_counter_local = trial_length  # from the buffer
for trial in range(num_trials):
    obs = env.reset()
    agent.reset()

    done = False
    total_reward = 0.0
    steps_trial = 0
    # update_axes(
    #     axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial,
    #     all_rewards
    # )
    episode_length = 0
    single_rewards = []
    eval_metric = []  # vel for cartpole and div for quad
    while not done:
        # --------------- Model Training -----------------
        if steps_trial == 0:
            # tic_dyn = time.time()
            if model_load_path is None:
                print("init normalizer")
                # only normalize if no normalizer was loaded
                dynamics_model.update_normalizer(
                    replay_buffer.get_all()
                )  # update normalizer stats

            dataset_train, dataset_val = common_util.get_basic_buffer_iterators(
                replay_buffer,
                batch_size=cfg.overrides.model_batch_size,
                val_ratio=cfg.overrides.validation_ratio,
                ensemble_size=ensemble_size,
                shuffle_each_epoch=True,
                bootstrap_permutes=
                False,  # build bootstrap dataset using sampling with replacement
            )

            model_trainer.train(
                dataset_train,
                dataset_val=dataset_val,
                num_epochs=50,
                patience=50,
                callback=train_callback
            )
            # print(steps_trial, "train dyn", time.time() - tic_dyn)
        step_counter_local += 1

        # episode length counter independent of rewards
        episode_length += 1
        # --- Doing env step using the agent and adding to model dataset ---
        # tic_act = time.time()
        next_obs, reward, done, _ = common_util.step_env_and_add_to_buffer(
            env, obs, agent, {}, replay_buffer
        )
        # print("action out time", time.time() - tic_act)

        # update_axes(
        #     axs, env.render(mode="rgb_array"), ax_text, trial, steps_trial,
        #     all_rewards
        # )
        # print("reward", reward, "vel", env.state[1])
        if SYSTEM == "cartpole":
            eval_metric.append(env.state[1])
        elif SYSTEM == "quad":
            eval_metric.append(env.get_divergence())
            print("div", eval_metric[-1])
        elif SYSTEM == "fixed_wing":
            eval_metric.append(env.get_divergence())
            print("div", eval_metric[-1])
        elif SYSTEM == "halfcheetah":
            reward = float(reward)
            eval_metric.append(reward)
            print("rew", reward)
        single_rewards.append(reward)
        obs = next_obs
        total_reward += reward
        steps_trial += 1

        if steps_trial == trial_length:
            break
    if SYSTEM == "fixed_wing":
        div_to_target.append(env.get_target_div())
        print("div to target", div_to_target[-1])
    eval_metric_list.append(float(np.mean(np.absolute(np.array(eval_metric)))))
    if SYSTEM == "halfcheetah":
        step_counter_list.append(step_counter_local)
    else:
        step_counter_list.append(env.step_counter)
    episode_lens_list.append(episode_length)
    print("total", total_reward)
    print("episode length", episode_length)
    print()
    sum_rewards.append(total_reward)
    all_rewards.append(single_rewards)
    timestamps.append(time.time())

    if trial == 0 or (trial + 1) % 5 == 0:
        episode_out_dir = f"eps_{trial}_{step_counter_list[-1]}"
        out_path = os.path.join(model_save_path, episode_out_dir)
        os.makedirs(out_path, exist_ok=True)
        dynamics_model.save(save_dir=out_path)
        make_dict_and_save()

# Save training results as json

fig, ax = plt.subplots(2, 1, figsize=(12, 10))
ax[0].plot(train_losses)
ax[0].set_xlabel("Total training epochs")
ax[0].set_ylabel("Training loss (avg. NLL)")
ax[1].plot(val_scores)
ax[1].set_xlabel("Total training epochs")
ax[1].set_ylabel("Validation score (avg. MSE)")
plt.savefig("../plot_mbrl_firsttry.png")
plt.show()

# # Where to learn more about MBRL?
