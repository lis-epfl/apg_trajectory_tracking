import os
import json
import time
import numpy as np
import torch
import torch.nn.functional as F

from neural_control.dataset import QuadDataset
from train_base import TrainBase
from neural_control.drone_loss import quad_mpc_loss
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamics
from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamics
)
from neural_control.dynamics.quad_dynamics_trained import LearntDynamics
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.environments.drone_env import QuadRotorEnvBase
from evaluate_drone import QuadEvaluator
from neural_control.models.hutter_model import Net
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass


class TrainDrone(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(self, train_dynamics, eval_dynamics, config):
        """
        param sample_in: one of "train_env", "eval_env", "real_flightmare"
        """
        self.config = config
        super().__init__(train_dynamics, eval_dynamics, **config)

        # Create environment for evaluation
        if self.sample_in == "real_flightmare":
            self.eval_env = FlightmareWrapper(self.delta_t)
        elif self.sample_in == "eval_env":
            self.eval_env = QuadRotorEnvBase(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = QuadRotorEnvBase(self.train_dynamics, self.delta_t)
        else:
            raise ValueError(
                "sample in must be one of eval_env, train_env, real_flightmare"
            )

    def initialize_model(
        self,
        base_model=None,
        modified_params={},
        base_model_name="model_quad"
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
            data_std = np.array(previous_parameters["std"]).astype(float)
            data_mean = np.array(previous_parameters["mean"]).astype(float)
        else:
            self.state_data = QuadDataset(
                self.epoch_size,
                self.self_play,
                reset_strength=self.reset_strength,
                max_drone_dist=self.max_drone_dist,
                ref_length=self.nr_actions,
                dt=self.delta_t
            )
            in_state_size = self.state_data.normed_states.size()[1]
            # +9 because adding 12 things but deleting position (3)
            self.net = Net(
                in_state_size,
                self.nr_actions,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=1
            )
            (data_std, data_mean) = (self.state_data.std, self.state_data.mean)

        # save std for normalization during test time
        self.config["std"] = data_std.tolist()
        self.config["mean"] = data_mean.tolist()

        # update the used parameters:
        self.config["horizon"] = self.nr_actions
        self.config["ref_length"] = self.nr_actions
        self.config["thresh_div"] = self.thresh_div_start
        self.config["dt"] = self.delta_t
        self.config["take_every_x"] = self.self_play_every_x
        self.config["thresh_stable"] = self.thresh_stable_start
        for k, v in modified_params.items():
            if type(v) == np.ndarray:
                modified_params[k] = v.tolist()
        self.config["modified_params"] = modified_params

        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        # init dataset
        self.state_data = QuadDataset(self.epoch_size, **self.config)
        self.init_optimizer()

    def train_controller_model(
        self, current_state, action_seq, in_ref_states, ref_states
    ):
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        # save the reached states
        intermediate_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(self.nr_actions):
            # extract action
            action = action_seq[:, k]
            current_state = self.train_dynamics(
                current_state, action, dt=self.delta_t
            )
            intermediate_states[:, k] = current_state

        loss = quad_mpc_loss(
            intermediate_states, ref_states, action_seq, printout=0
        )

        # Backprop
        loss.backward()
        self.writer.add_scalar('loss/training', loss)
        # for name, param in self.net.named_parameters():
        #     if param.grad is not None:
        #         self.writer.add_histogram(name + ".grad", param.grad)
        self.optimizer_controller.step()
        return loss

    def evaluate_model(self, epoch):
        # EVALUATE
        controller = NetworkWrapper(self.net, self.state_data, **self.config)

        evaluator = QuadEvaluator(controller, self.eval_env, **self.config)
        # run with mpc to collect data
        # eval_env.run_mpc_ref("rand", nr_test=5, max_steps=500)
        # run without mpc for evaluation
        with torch.no_grad():
            suc_mean, suc_std, div_full_mean, div_full_std, div_mean, div_std = evaluator.run_eval(
                "rand", nr_test=10, **self.config
            )

        self.sample_new_data(epoch)

        # increase threshold
        if epoch % 5 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += .05
            print(
                "Curriculum learning: increase divergence threshold",
                round(self.config["thresh_div"], 2)
            )

        # save best model
        self.save_model(epoch, suc_mean, suc_std)

        self.results_dict["mean_divergence_full"].append(div_full_mean)
        self.results_dict["std_divergence_full"].append(div_full_std)
        self.results_dict["mean_divergence"].append(div_mean)
        self.results_dict["std_divergence"].append(div_std)
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
    print(modified_params)
    train_dynamics = FlightmareDynamics(modified_params=modified_params)
    eval_dynamics = FlightmareDynamics(modified_params=modified_params)

    # make sure that also the self play samples are collected in same env
    config["sample_in"] = "train_env"

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    trainer.run_control(config)


def train_dynamics(base_model, config):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"

    # train environment is learnt
    train_dynamics = LearntDynamics()
    eval_dynamics = FlightmareDynamics(modified_params)

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
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
    train_dynamics = FlightmareDynamics()
    eval_dynamics = FlightmareDynamics(modified_params=modified_params)

    trainer = TrainDrone(train_dynamics, eval_dynamics, config)
    trainer.initialize_model(base_model, modified_params=modified_params)

    # RUN
    trainer.run_control(config, sampling_based_finetune=True)


if __name__ == "__main__":
    # LOAD CONFIG
    with open("configs/quad_config.json", "r") as infile:
        config = json.load(infile)

    # mod_params = {"mass": 1}
    # # {'translational_drag': np.array([0.7, 0.7, 0.7])}
    # config["modified_params"] = mod_params

    baseline_model = None  # "trained_models/quad/"
    # config["thresh_div_start"] = 1
    # config["thresh_stable_start"] = 1.5

    config["save_name"] = "mpc_loss"

    config["nr_epochs"] = 400

    # TRAIN
    train_control(baseline_model, config)
    # train_dynamics(baseline_model, config)
    # train_sampling_finetune(baseline_model, config)
    # FINE TUNING parameters:
    # self.thresh_div_start = 1
    # self.self_play = 1.5
    # self.epoch_size = 500
    # self.max_steps = 1000
    # self.self_play_every_x = 5
    # self.learning_rate = 0.0001
