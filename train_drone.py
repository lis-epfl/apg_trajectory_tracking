import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from neural_control.dataset import DroneDataset
from neural_control.drone_loss import (
    drone_loss_function, simply_last_loss, reference_loss, mse_loss,
    weighted_loss
)
from neural_control.dynamics.quad_dynamics_simple import SimpleDynamics
from neural_control.dynamics.quad_dynamics_flightmare import (
    FlightmareDynamics
)
from neural_control.dynamics.quad_dynamics_trained import LearntDynamics
from neural_control.controllers.network_wrapper import NetworkWrapper
from neural_control.environments.drone_env import QuadRotorEnvBase
from evaluate_drone import QuadEvaluator
from neural_control.models.hutter_model import Net
from neural_control.plotting import (
    plot_loss_episode_len, print_state_ref_div
)
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass


class TrainDrone:
    """
    Train a controller for a quadrotor
    """

    def __init__(
        self,
        train_dynamics,
        eval_dynamics,
        speed_factor=.6,
        sample_in="train_env"
    ):
        """
        param sample_in: one of "train_env", "eval_env", "real_flightmare"
        """
        self.sample_in = sample_in
        self.delta_t = 0.1
        self.epoch_size = 500
        self.self_play = 1
        self.self_play_every_x = 2
        self.batch_size = 8
        self.reset_strength = 1.2
        self.max_drone_dist = 0.25
        self.thresh_div_start = .1
        self.thresh_div_end = 2
        self.thresh_stable = 1.5
        self.state_size = 12
        self.nr_actions = 10
        self.ref_dim = 9
        self.action_dim = 4
        self.learning_rate_controller = 0.0001
        self.learning_rate_dynamics = 0.001
        self.speed_factor = speed_factor
        self.max_steps = 1000
        self.save_path = os.path.join("trained_models/drone/test_model")

        # initialize as the original flightmare environment
        self.train_dynamics = train_dynamics
        # FINE TUNING:
        # self.thresh_div_start = 1
        # self.self_play = 1.5
        # self.epoch_size = 500
        # self.max_steps = 1000
        # self.self_play_every_x = 5
        # self.learning_rate = 0.0001
        self.eval_dynamics = eval_dynamics

        self.count_finetune_data = 0
        self.highest_success = 0

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

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
            with open(
                os.path.join(base_model, "param_dict.json"), "r"
            ) as outfile:
                self.param_dict = json.load(outfile)
            STD = np.array(self.param_dict["std"]).astype(float)
            MEAN = np.array(self.param_dict["mean"]).astype(float)
        else:
            self.state_data = DroneDataset(
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
            (STD, MEAN) = (self.state_data.std, self.state_data.mean)

        # save std for normalization during test time
        self.param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
        # update the used parameters:
        self.param_dict["reset_strength"] = self.reset_strength
        self.param_dict["max_drone_dist"] = self.max_drone_dist
        self.param_dict["horizon"] = self.nr_actions
        self.param_dict["ref_length"] = self.nr_actions
        self.param_dict["thresh_div"] = self.thresh_div_start
        self.param_dict["dt"] = self.delta_t
        self.param_dict["take_every_x"] = self.self_play_every_x
        self.param_dict["thresh_stable"] = self.thresh_stable
        self.param_dict["speed_factor"] = self.speed_factor
        for k, v in modified_params.items():
            if type(v) == np.ndarray:
                modified_params[k] = v.tolist()
        self.param_dict["modified_params"] = modified_params

        with open(
            os.path.join(self.save_path, "param_dict.json"), "w"
        ) as outfile:
            json.dump(self.param_dict, outfile)

        # init dataset
        self.state_data = DroneDataset(
            self.epoch_size, self.self_play, **self.param_dict
        )
        # Init train loader
        self.trainloader = torch.utils.data.DataLoader(
            self.state_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        # init optimizer and torch normalization parameters
        self.optimizer_controller = optim.SGD(
            self.net.parameters(),
            lr=self.learning_rate_controller,
            momentum=0.9
        )
        if isinstance(self.train_dynamics, LearntDynamics):
            self.optimizer_dynamics = optim.SGD(
                self.train_dynamics.parameters(),
                lr=self.learning_rate_dynamics,
                momentum=0.9
            )

    def train_controller_model(self, current_state, action_seq, ref_states):
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        # save the reached states
        intermediate_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(self.nr_actions):
            # extract action
            action = action_seq[:, k]
            current_state = self.train_dynamics.simulate_quadrotor(
                action, current_state, dt=self.delta_t
            )
            intermediate_states[:, k] = current_state

        loss = simply_last_loss(
            intermediate_states, ref_states[:, -1], action_seq, printout=0
        )

        # Backprop
        loss.backward()
        self.optimizer_controller.step()
        return loss

    def train_dynamics_model(self, current_state, action_seq):
        # zero the parameter gradients
        self.optimizer_dynamics.zero_grad()
        next_state_d1 = self.train_dynamics(
            action_seq[:, 0], current_state, dt=self.delta_t
        )
        next_state_d2 = self.eval_dynamics.simulate_quadrotor(
            action_seq[:, 0], current_state, dt=self.delta_t
        )
        # TODO: weighting --> now velocity much more than attitude etc
        loss = torch.sum((next_state_d1 - next_state_d2)**2)
        # print(self.train_dynamics.down_drag.grad)
        # print("grad", self.train_dynamics.linear_state_2.weight.grad)
        loss.backward()
        self.optimizer_dynamics.step()
        return loss

    def run_epoch(self, train="controller"):
        # tic_epoch = time.time()
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, current_state, in_ref_state, ref_states = data

            # ------------ VERSION 1 (x states at once)-----------------
            actions = self.net(in_state, in_ref_state)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(
                actions, (-1, self.nr_actions, self.action_dim)
            )

            if train == "controller":
                loss = self.train_controller_model(
                    current_state, action_seq, ref_states
                )
            else:
                loss = self.train_dynamics_model(current_state, action_seq)
                self.count_finetune_data += len(current_state)

            running_loss += loss.item() * 1000
        # time_epoch = time.time() - tic
        return running_loss / i

    def evaluate_model(self, epoch):
        # EVALUATE
        print(f"Epoch {epoch} (before)")
        controller = NetworkWrapper(
            self.net, self.state_data, **self.param_dict
        )
        # specify  self.sample_in to collect more data (exploration)
        if self.sample_in == "real_flightmare":
            eval_env = FlightmareWrapper(self.param_dict["dt"])
        elif self.sample_in == "eval_env":
            eval_env = QuadRotorEnvBase(
                self.eval_dynamics, self.param_dict["dt"]
            )
        elif self.sample_in == "train_env":
            eval_env = QuadRotorEnvBase(
                self.train_dynamics, self.param_dict["dt"]
            )
        else:
            raise ValueError(
                "sample in must be one of eval_env, train_env, real_flightmare"
            )

        evaluator = QuadEvaluator(controller, eval_env, **self.param_dict)
        # run with mpc to collect data
        # eval_env.run_mpc_ref("rand", nr_test=5, max_steps=500)
        # run without mpc for evaluation
        with torch.no_grad():
            suc_mean, suc_std = evaluator.eval_ref(
                "rand",
                nr_test=10,
                max_steps=self.max_steps,
                **self.param_dict
            )

        if (epoch + 1) % 3 == 0:
            # renew the sampled data
            self.state_data.resample_data()
            print(f"Sampled new data ({self.state_data.num_sampled_states})")

        if epoch % 5 == 0 and self.param_dict["thresh_div"
                                              ] < self.thresh_div_end:
            self.param_dict["thresh_div"] += .05
            print("increased thresh div", self.param_dict["thresh_div"])

        # save best model
        if epoch > 0 and suc_mean > self.highest_success:
            self.highest_success = suc_mean
            print("Best model")
            torch.save(
                self.net,
                os.path.join(self.save_path, "model_quad" + str(epoch))
            )

        return suc_mean, suc_std

    def finalize(self, mean_list, std_list, losses):
        torch.save(self.net, os.path.join(self.save_path, "model_quad"))
        plot_loss_episode_len(
            mean_list,
            std_list,
            losses,
            save_path=os.path.join(self.save_path, "performance.png")
        )
        print("finished and saved.")


if __name__ == "__main__":

    method = "train_control"
    modified_params = {"translational_drag": np.array([.3, .3, .3])}
    base_model = "trained_models/drone/baseline_flightmare"

    if method == "sample":
        nr_epochs = 10
        train_dyn = -1
        sample_in = "eval_env"
    elif method == "train_dyn":
        nr_epochs = 200
        train_dyn = 10
        sample_in = "train_env"
    elif method == "train_control":
        nr_epochs = 200
        train_dyn = -1
        sample_in = "train_env"

    train_dynamics = FlightmareDynamics()
    # LearntDynamics()

    eval_dynamics = None  # FlightmareDynamics(**modified_params)

    trainer = TrainDrone(train_dynamics, eval_dynamics, sample_in=sample_in)

    trainer.initialize_model(base_model, modified_params=modified_params)

    loss_list, success_mean_list, success_std_list = list(), list(), list()

    try:
        for epoch in range(nr_epochs):
            model_to_train = "dynamics" if epoch < train_dyn else "controller"

            if epoch == train_dyn:
                print("Params of dynamics model after training:")
                for k, v in trainer.train_dynamics.state_dict().items():
                    print(k, v)

            # EVALUATE
            suc_mean, suc_std = trainer.evaluate_model(epoch)
            success_mean_list.append(suc_mean)
            success_std_list.append(suc_std)

            print("Data used to train dynamics:", trainer.count_finetune_data)
            print(
                "Sampled data (exploration):", trainer.state_data.eval_counter
            )
            print()

            # RUN training
            epoch_loss = trainer.run_epoch(train=model_to_train)

            loss_list.append(epoch_loss)

            print(f"Loss ({model_to_train}): {round(epoch_loss, 2)}")
            # print("time one epoch", time.time() - tic_epoch)
    except KeyboardInterrupt:
        # Save model
        trainer.finalize(success_mean_list, success_std_list, loss_list)
