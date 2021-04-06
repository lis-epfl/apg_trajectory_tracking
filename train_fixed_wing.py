import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from neural_control.dataset import WingDataset
from neural_control.drone_loss import (
    trajectory_loss, fixed_wing_loss, angle_loss
)
from neural_control.dynamics.fixed_wing_dynamics import (
    FixedWingDynamics, LearntFixedWingDynamics
)
from neural_control.environments.wing_env import SimpleWingEnv
from neural_control.models.hutter_model import Net
from evaluate_fixed_wing import FixedWingEvaluator
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.plotting import plot_loss_episode_len


class TrainFixedWing:
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
        param sample_in: one of "train_env", "eval_env"
        """
        self.sample_in = sample_in
        self.delta_t = 0.05
        self.epoch_size = 1000
        self.vec_std = 0.15
        self.self_play = .5
        self.self_play_every_x = 2
        self.batch_size = 8
        self.reset_strength = 1.2
        self.max_drone_dist = 0.25
        self.thresh_div_start = 4
        self.thresh_div_end = 5
        self.thresh_stable_start = .4
        self.thresh_stable_end = .8
        self.state_size = 12
        self.nr_actions = 10
        self.nr_actions_rnn = 10
        self.ref_dim = 3
        self.action_dim = 4
        self.learning_rate_controller = 0.00001
        self.learning_rate_dynamics = 0.001
        self.save_path = os.path.join("trained_models/wing/test_model")

        # initialize as the original flightmare environment
        self.train_dynamics = train_dynamics
        self.eval_dynamics = eval_dynamics

        self.count_finetune_data = 0
        self.highest_success = np.inf

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # specify  self.sample_in to collect more data (exploration)
        if self.sample_in == "eval_env":
            self.eval_env = SimpleWingEnv(
                self.eval_dynamics, self.param_dict["dt"]
            )
        elif self.sample_in == "train_env":
            self.eval_env = SimpleWingEnv(
                self.train_dynamics, self.param_dict["dt"]
            )
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
            with open(
                os.path.join(base_model, "param_dict.json"), "r"
            ) as outfile:
                self.param_dict = json.load(outfile)
        else:
            self.param_dict = {
                "dt": self.delta_t,
                "horizon": self.nr_actions,
                "vec_std": self.vec_std
            }
            # +9 because adding 12 things but deleting position (3)
            self.net = Net(
                self.state_size - self.ref_dim,
                1,
                self.ref_dim,
                self.action_dim * self.nr_actions,
                conv=False
            )

        # init dataset
        self.state_data = WingDataset(
            self.epoch_size, self_play=self.self_play, **self.param_dict
        )
        self.param_dict = self.state_data.get_means_stds(self.param_dict)
        self.param_dict["take_every_x"] = self.self_play_every_x
        self.param_dict["thresh_div"] = self.thresh_div_start
        self.param_dict["thresh_stable"] = self.thresh_stable_start

        with open(
            os.path.join(self.save_path, "param_dict.json"), "w"
        ) as outfile:
            json.dump(self.param_dict, outfile)

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
        if isinstance(self.train_dynamics, LearntFixedWingDynamics):
            self.optimizer_dynamics = optim.SGD(
                self.train_dynamics.parameters(),
                lr=self.learning_rate_dynamics,
                momentum=0.9
            )

    def _compute_target_pos(self, current_state, ref_vector):
        # # GIVE LINEAR TRAJECTORY FOR LOSS
        speed = torch.sqrt(torch.sum(current_state[:, 3:6]**2, dim=1))
        vec_len_per_step = speed * self.delta_t * self.nr_actions_rnn
        # form auxiliary array with linear reference for loss computation
        target_pos = torch.zeros((current_state.size()[0], 3))
        for j in range(3):
            target_pos[:, j] = current_state[:, j] + (
                ref_vector[:, j] * vec_len_per_step
            )
        return target_pos

    def train_controller_model(self, current_state, target_pos, action_seq):
        # zero the parameter gradients
        self.optimizer_controller.zero_grad()
        # intermediate_states = torch.zeros(
        #     in_state.size()[0], NR_ACTIONS,
        #     current_state.size()[1]
        # )
        for k in range(self.nr_actions_rnn):
            # extract action
            action = action_seq[:, k]
            current_state = self.train_dynamics.simulate_fixed_wing(
                current_state, action, dt=self.delta_t
            )
            # intermediate_states[:, k] = current_state

        loss = fixed_wing_loss(
            current_state, target_pos, action_seq, printout=0
        )

        # Backprop
        loss.backward(retain_graph=1)  # TODO necessary???
        self.optimizer_controller.step()
        return loss

    def train_controller_recurrent(self, current_state, ref_state, target_pos):
        # ------------ VERSION 2: recurrent -------------------
        for k in range(self.nr_actions_rnn):
            in_state, _, in_ref_state, _ = self.state_data.prepare_data(
                current_state, ref_state
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

    def train_dynamics_model(self, current_state, action_seq):
        # TODO: This is the same for both fixed wing and quad --> inherit?
        # zero the parameter gradients
        self.optimizer_dynamics.zero_grad()
        next_state_d1 = self.train_dynamics(
            action_seq[:, 0], current_state, dt=self.delta_t
        )
        next_state_d2 = self.eval_dynamics.simulate_fixed_wing(
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

            actions = self.net(in_state, in_ref_state)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(
                actions, (-1, self.nr_actions, self.action_dim)
            )

            if train == "controller":
                target_pos = self._compute_target_pos(
                    current_state, in_ref_state
                )
                loss = self.train_controller_model(
                    current_state, target_pos, action_seq
                )
                # # ---- recurrent --------
                # loss = self.train_controller_recurrent(
                #     current_state, ref_states, target_pos
                # )
            else:
                # should work for both recurrent and normal
                loss = self.train_dynamics_model(current_state, action_seq)
                self.count_finetune_data += len(current_state)

            running_loss += loss.item() * 1000
        # time_epoch = time.time() - tic
        return running_loss / i

    def evaluate_model(self, epoch):
        # EVALUATE
        print(f"Epoch {epoch} (before)")
        controller = FixedWingNetWrapper(
            self.net, self.state_data, **self.param_dict
        )

        evaluator = FixedWingEvaluator(
            controller, self.eval_env, **self.param_dict
        )
        # run with mpc to collect data
        # eval_env.run_mpc_ref("rand", nr_test=5, max_steps=500)
        # run without mpc for evaluation
        with torch.no_grad():
            suc_mean, suc_std = evaluator.run_eval(nr_test=10)

        if (epoch + 1) % 3 == 0:
            # renew the sampled data
            self.state_data.resample_data()
            print(f"Sampled new data ({self.state_data.num_sampled_states})")

        if epoch % 5 == 0 and self.param_dict["thresh_div"
                                              ] < self.thresh_div_end:
            self.param_dict["thresh_div"] += .05
            print("increased thresh div", self.param_dict["thresh_div"])

        # save best model
        if epoch > 0 and suc_mean < self.highest_success:
            self.highest_success = suc_mean
            print("Best model")
            torch.save(
                self.net,
                os.path.join(self.save_path, "model_wing" + str(epoch))
            )

        return suc_mean, suc_std

    def finalize(self, mean_list, std_list, losses):
        torch.save(self.net, os.path.join(self.save_path, "model_wing"))
        plot_loss_episode_len(
            mean_list,
            std_list,
            losses,
            save_path=os.path.join(self.save_path, "performance.png")
        )
        print("finished and saved.")


if __name__ == "__main__":

    method = "train_control"
    modified_params = {}
    base_model = None  # "trained_models/wing/best_23"

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

    train_dynamics = LearntFixedWingDynamics(modified_params)

    eval_dynamics = None

    trainer = TrainFixedWing(
        train_dynamics, eval_dynamics, sample_in=sample_in
    )

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

            if epoch < train_dyn:
                print(
                    "Data used to train dynamics:", trainer.count_finetune_data
                )
            if sample_in == "eval_env":
                print(
                    "Sampled data (exploration):",
                    trainer.state_data.eval_counter
                )
            if method == "train_control":
                print(
                    "Data used for training:",
                    epoch * (trainer.epoch_size * (1 + trainer.self_play))
                )

            # RUN training
            epoch_loss = trainer.run_epoch(train=model_to_train)

            loss_list.append(epoch_loss)

            print()
            print(f"Loss ({model_to_train}): {round(epoch_loss, 2)}")
            # print("time one epoch", time.time() - tic_epoch)
    except KeyboardInterrupt:
        pass
    # Save model
    trainer.finalize(success_mean_list, success_std_list, loss_list)
