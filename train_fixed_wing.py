import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from neural_control.dataset import WingDataset
from neural_control.drone_loss import (
    trajectory_loss, fixed_wing_loss, angle_loss, fixed_wing_mpc_loss
)
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, LearntFixedWingDynamics
)
from neural_control.environments.wing_env import SimpleWingEnv
from neural_control.models.hutter_model import Net
from evaluate_fixed_wing import FixedWingEvaluator
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from train_base import TrainBase


class TrainFixedWing(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(self, train_dynamics, eval_dynamics, config):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.config = config
        super().__init__(train_dynamics, eval_dynamics, **config)

        # specify  self.sample_in to collect more data (exploration)
        if self.sample_in == "eval_env":
            self.eval_env = SimpleWingEnv(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = SimpleWingEnv(self.train_dynamics, self.delta_t)
        else:
            raise ValueError("sample in must be one of eval_env, train_env")

    def initialize_model(
        self,
        base_model=None,
        modified_params={},
        base_model_name="model_wing"
    ):
        # Load model or initialize model
        if base_model is not None:
            self.net = torch.load(os.path.join(base_model, base_model_name))
            # load std or other parameters from json
            config_path = os.path.join(base_model, "config.json")
            if not os.path.exists(config_path):
                print("Load old config..")
                config_path = os.path.join(base_model, "param_dict.json")
            with open(config_path, "r") as outfile:
                previous_parameters = json.load(outfile)
                self.config["mean"] = previous_parameters["mean"]
                self.config["std"] = previous_parameters["std"]
        else:
            # +9 because adding 12 things but deleting position (3)
            self.net = Net(
                self.state_size - self.ref_dim,
                1,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=False
            )

        # init dataset
        self.state_data = WingDataset(self.epoch_size, **self.config)
        # update mean and std:
        self.config = self.state_data.get_means_stds(self.config)
        # add other parameters
        self.config["horizon"] = self.nr_actions
        self.config["ref_length"] = self.nr_actions
        self.config["thresh_div"] = self.thresh_div_start
        self.config["dt"] = self.delta_t
        self.config["take_every_x"] = self.self_play_every_x
        self.config["thresh_stable"] = self.thresh_stable_start

        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        self.init_optimizer()

    def _compute_target_pos(self, current_state, ref_vector):
        # # GIVE LINEAR TRAJECTORY FOR LOSS
        # speed = torch.sqrt(torch.sum(current_state[:, 3:6]**2, dim=1))
        vec_len_per_step = 12 * self.delta_t_train
        # form auxiliary array with linear reference for loss computation
        target_pos = torch.zeros(
            (current_state.size()[0], self.nr_actions_rnn, 3)
        )
        for i in range(self.nr_actions_rnn):
            for j in range(3):
                target_pos[:, i, j] = current_state[:, j] + (
                    ref_vector[:, j] * vec_len_per_step * (i + 1)
                )
        return target_pos

    def train_controller_model(
        self, current_state, action_seq, in_ref_state, ref_states
    ):
        target_pos = self._compute_target_pos(current_state, in_ref_state)
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        intermediate_states = torch.zeros(
            current_state.size()[0], self.nr_actions_rnn, self.state_size
        )
        for k in range(self.nr_actions_rnn):
            # extract action
            action = action_seq[:, k]
            current_state = self.train_dynamics(
                current_state, action, dt=self.delta_t_train
            )
            intermediate_states[:, k] = current_state

        loss = fixed_wing_mpc_loss(
            intermediate_states, target_pos, action_seq, printout=0
        )

        # Backprop
        loss.backward()
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(name + ".grad", param.grad)
        self.optimizer_controller.step()
        return loss

    def train_controller_recurrent(
        self, current_state, action_seq, in_ref_state, ref_states
    ):
        target_pos = self._compute_target_pos(current_state, in_ref_state)
        # ------------ VERSION 2: recurrent -------------------
        for k in range(self.nr_actions_rnn):
            in_state, _, in_ref_state, _ = self.state_data.prepare_data(
                current_state, ref_states
            )
            # print(k, "current state", current_state[0, :3])
            # print(k, "in_state", in_state[0])
            # print(k, "in ref", in_ref_state[0])
            action = torch.sigmoid(self.net(in_state, in_ref_state))
            current_state = self.train_dynamics.simulate_fixed_wing(
                current_state, action, dt=self.delta_t
            )
        loss = fixed_wing_loss(current_state, target_pos, None, printout=0)
        # Backprop
        loss.backward()
        self.optimizer_controller.step()
        return loss

    def evaluate_model(self, epoch):
        # EVALUATE
        controller = FixedWingNetWrapper(
            self.net, self.state_data, **self.config
        )

        evaluator = FixedWingEvaluator(
            controller, self.eval_env, **self.config
        )
        # run with mpc to collect data
        # eval_env.run_mpc_ref("rand", nr_test=5, max_steps=500)
        # run without mpc for evaluation
        with torch.no_grad():
            suc_mean, suc_std = evaluator.run_eval(nr_test=10)

        self.sample_new_data(epoch)

        # increase thresholds
        if epoch % 5 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += .2
            print("increased thresh div", self.config["thresh_div"])

        if epoch % 5 == 0 and self.config["thresh_stable"
                                          ] < self.thresh_stable_end:
            self.config["thresh_stable"] += .05

        # save best model
        self.save_model(epoch, suc_mean, suc_std)

        self.results_dict["mean_success"].append(suc_mean)
        self.results_dict["std_success"].append(suc_std)
        self.results_dict["thresh_div"].append(self.config["thresh_div"])
        return suc_mean, suc_std


def train_control(base_model, config):
    """
    Train a controller from scratch or with an initial model
    """
    modified_params = config["modified_params"]
    # TODO: might be problematic
    train_dynamics = FixedWingDynamics(modified_params)
    eval_dynamics = FixedWingDynamics(modified_params)

    # make sure that also the self play samples are collected in same env
    config["sample_in"] = "train_env"

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    trainer.run_control(config, curriculum=0)


def train_dynamics(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"
    # set thresholds high so the tracking error is reliable
    config["thresh_div_start"] = 20
    config["thresh_stable_start"] = 1.5

    # train environment is learnt
    train_dynamics = LearntFixedWingDynamics()
    eval_dynamics = FixedWingDynamics(modified_params=modified_params)

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_dynamics(config)


def train_sampling_finetune(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "eval_env"

    # train environment is learnt
    train_dynamics = FixedWingDynamics()
    eval_dynamics = FixedWingDynamics(modified_params=modified_params)

    trainer = TrainFixedWing(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_control(config, sampling_based_finetune=True)


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/wing_config.json", "r") as infile:
        config = json.load(infile)

    baseline_model = None  # "trained_models/wing/baseline_fixed_wing"
    config["save_name"] = "train_mpc_loss"

    # set high thresholds because not training from scratch
    # config["thresh_div_start"] = 20
    # config["thresh_stable_start"] = 1.5

    mod_params = {}
    #  {"rho": 1.6}
    # {
    #     "CL0": 0.3,  # 0.39
    #     "CD0": 0.02,  #  0.0765,
    #     "CY0": 0.02,  # 0.0,
    #     "Cl0": -0.01,  # 0.0,
    #     "Cm0": 0.01,  # 0.02,
    #     "Cn0": 0.0,
    # }
    # # {"rho": 1.4, "mass": 1.2, "S": 0.32}  # mass: 1.4
    config["modified_params"] = mod_params

    # TRAIN
    # config["nr_epochs"] = 20
    train_control(baseline_model, config)
    # train_dynamics(baseline_model, config)
    # train_sampling_finetune(baseline_model, config)
