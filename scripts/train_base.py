import os
import json
import time
import numpy as np
import torch
import torch.optim as optim
from collections import defaultdict
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:
        def __init__(self):
            print("Tensorboard not installed, not logging")
            pass
        def close(self):
            pass
        def flush(self):
            pass
        def add_scalar(self, name, scalar):
            pass
        def add_histogram(self, name, data):
            pass
from neural_control.dynamics.quad_dynamics_trained import LearntDynamics
from neural_control.dynamics.fixed_wing_dynamics import LearntFixedWingDynamics
from neural_control.plotting import (
    plot_loss_episode_len, print_state_ref_div
)


class TrainBase:

    def __init__(
        self,
        train_dynamics,
        eval_dynamics,
        sample_in="train_env",
        delta_t=0.05,
        delta_t_train=0.05,
        epoch_size=500,
        vec_std=0.15,
        self_play=1.5,
        self_play_every_x=2,
        batch_size=8,
        reset_strength=1.2,
        max_drone_dist=0.25,
        max_steps=1000,
        thresh_div_start=4,
        thresh_div_end=20,
        thresh_stable_start=.4,
        thresh_stable_end=.8,
        state_size=12,
        nr_actions=10,
        nr_actions_rnn=10,
        ref_dim=3,
        action_dim=4,
        l2_lambda=0.1,
        learning_rate_controller=0.0001,
        learning_rate_dynamics=0.001,
        speed_factor=.6,
        resample_every=3,
        suc_up_down=1,
        train_mode="concurrent", 
        system="quad",
        save_name="test_model",
        **kwargs
    ):
        self.sample_in = sample_in
        self.delta_t = delta_t
        self.delta_t_train = delta_t_train
        self.epoch_size = epoch_size
        self.vec_std = vec_std
        self.self_play = self_play
        self.self_play_every_x = self_play_every_x
        self.batch_size = batch_size
        self.reset_strength = reset_strength
        self.max_drone_dist = max_drone_dist
        self.thresh_div_start = thresh_div_start
        self.thresh_div_end = thresh_div_end
        self.thresh_stable_start = thresh_stable_start
        self.thresh_stable_end = thresh_stable_end
        self.state_size = state_size
        self.nr_actions = nr_actions
        self.nr_actions_rnn = nr_actions_rnn
        self.ref_dim = ref_dim
        self.action_dim = action_dim
        self.l2_lambda = l2_lambda
        self.speed_factor = speed_factor
        self.max_steps = max_steps
        self.resample_every = resample_every
        self.suc_up_down = suc_up_down
        self.learning_rate_controller = learning_rate_controller
        self.learning_rate_dynamics = learning_rate_dynamics
        self.train_mode = train_mode

        # performance logging:
        self.results_dict = defaultdict(list)
        # to match losses and eval runs, add 0
        self.results_dict["loss"].append(0)

        # saving routine
        self.save_name = save_name
        self.save_path = os.path.join("trained_models", system, save_name)
        self.save_model_name = "model_" + system
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        else:
            input("Save directory already exists. Continue?")

        # dynamics
        self.eval_dynamics = eval_dynamics
        self.train_dynamics = train_dynamics

        self.count_finetune_data = 0
        self.sampled_data_count = 0

        self.current_score = 0 if suc_up_down == 1 else np.inf

        self.state_data = None
        self.net = None

        self.writer = SummaryWriter()

    def init_optimizer(self):
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
        if isinstance(self.train_dynamics, LearntFixedWingDynamics
                      ) or isinstance(self.train_dynamics, LearntDynamics):
            self.optimizer_dynamics = optim.SGD(
                self.train_dynamics.parameters(),
                lr=self.learning_rate_dynamics,
                momentum=0.9
            )

    def train_controller_model(
        self, current_state, action_seq, in_ref_state, ref_states
    ):
        """
        Implemented in sub classes
        """
        pass

    def train_dynamics_model(self, current_state, action_seq):
        # zero the parameter gradients
        self.optimizer_dynamics.zero_grad()
        next_state_d1 = self.train_dynamics(
            current_state, action_seq[:, 0], dt=self.delta_t
        )
        next_state_d2 = self.eval_dynamics(
            current_state, action_seq[:, 0], dt=self.delta_t
        )
        # regularize:
        l2_loss = 0
        if self.l2_lambda > 0:
            l2_loss = (
                torch.norm(self.train_dynamics.linear_state_2.weight) +
                torch.norm(self.train_dynamics.linear_state_2.bias) +
                torch.norm(self.train_dynamics.linear_state_1.weight) +
                torch.norm(self.train_dynamics.linear_state_1.bias)
            )
        # TODO: weighting --> now velocity much more than attitude etc
        loss = torch.sum(
            (next_state_d1 - next_state_d2)**2
        ) + self.l2_lambda * l2_loss
        loss.backward()
        self.optimizer_dynamics.step()

        self.results_dict["loss_dyn_per_step"].append(loss.item())
        return loss

    def run_epoch(self, train="controller", epoch=0):
        # tic_epoch = time.time()
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, current_state, in_ref_state, ref_states = data

            if self.train_mode != "concurrent":
                # if recurrent or recurrent_LSTM, use this one
                loss = self.train_recurrent_model(
                        in_state, current_state, in_ref_state, ref_states
                    )
            else:
                actions = self.net(in_state, in_ref_state)
                actions = torch.sigmoid(actions)
                action_seq = torch.reshape(
                    actions, (-1, self.nr_actions, self.action_dim)
                )
                loss = self.train_controller_model(
                    current_state, action_seq, in_ref_state, ref_states
                )

            running_loss += loss.item()
        # time_epoch = time.time() - tic
        epoch_loss = running_loss / i
        self.results_dict["loss"].append(epoch_loss)
        self.results_dict["trained"].append(train)
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        self.writer.add_scalar("Loss/train", epoch_loss, epoch)
        return epoch_loss

    def sample_new_data(self, epoch):
        """
        Every few epochs, resample data
        Args:
            epoch (int): Current epoch count
        """
        if (epoch + 1) % self.resample_every == 0:
            # renew the sampled data
            self.state_data.resample_data()
            # increase count
            self.sampled_data_count += self.state_data.num_sampled_states
            print(f"Sampled new data ({self.state_data.num_sampled_states})")

    def save_model(self, epoch, success, suc_std):
        # check if we either are higher than the current score (if measuring
        # the number of epochs) or lower (if measuring tracking error)
        if epoch > 0: # and (
            # success > self.current_score and self.suc_up_down == 1
        # ) or (success < self.current_score and self.suc_up_down == -1):
            self.current_score = success
            print("Save model with score ", round(success, 2))
            torch.save(
                self.net,
                os.path.join(
                    self.save_path, self.save_model_name + str(epoch)
                )
            )

        # In any case do tensorboard logging
        # for name, param in self.net.named_parameters():
        #     self.writer.add_histogram(name, param)
        self.writer.add_scalar("success_mean", success, epoch)
        self.writer.add_scalar("success_std", suc_std, epoch)
        self.writer.flush()

    def finalize(self):
        """
        Save model and plot loss and performance
        """
        torch.save(
            self.net, os.path.join(self.save_path, self.save_model_name)
        ) 
        np.savetxt(os.path.join(self.save_path, 'mean_successes.csv'),self.results_dict["mean_success"], delimiter=',')
        np.savetxt(os.path.join(self.save_path, 'std_success.csv'), self.results_dict["std_success"], delimiter=',')
        np.savetxt(os.path.join(self.save_path, 'loss.csv'), self.results_dict["loss"], delimiter=',')
        np.savetxt(os.path.join(self.save_path, 'mean_divergence_full.csv'), self.results_dict["mean_divergence_full"], delimiter=',')
        np.savetxt(os.path.join(self.save_path, 'std_divergence_full.csv'), self.results_dict["std_divergence_full"], delimiter=',')
        np.savetxt(os.path.join(self.save_path, 'mean_divergence.csv'), self.results_dict["mean_divergence"], delimiter=',')
        np.savetxt(os.path.join(self.save_path, 'std_divergence.csv'), self.results_dict["std_divergence"], delimiter=',')

        # plot performance
        plot_loss_episode_len(
            self.results_dict["mean_success"],
            self.results_dict["std_success"],
            self.results_dict["loss"],
            save_path=os.path.join(self.save_path, "performance.png")
        )
        # save performance logging
        with open(os.path.join(self.save_path, "results.json"), "w") as ofile:
            json.dump(self.results_dict, ofile)

        # save dynamics model if applicable
        if isinstance(self.train_dynamics, LearntFixedWingDynamics
                      ) or isinstance(self.train_dynamics, LearntDynamics):
            torch.save(
                self.train_dynamics.state_dict(),
                os.path.join(self.save_path, "dynamics_model")
            )
        self.writer.close()
        print("finished and saved.")

    def run_control(
        self, config, sampling_based_finetune=False, curriculum=1
    ):
        if curriculum:
            self.config["speed_factor"] = 0.2
            successes = []
        try:
            first_epoch_with_this_vel = 0
            for epoch in range(config["nr_epochs"]):
                _ = self.evaluate_model(epoch)

                if curriculum:
                    current_possible_steps = 1000 / (self.config["speed_factor"] / self.config["delta_t"])
                    successes.append(self.results_dict["mean_success"][-1])
                    print(successes, "speed", round(self.config["speed_factor"]
                        , 2), "thresh", round(self.config["thresh_div"], 2))
                    if ( (len(successes) > 5 and np.all(np.array(successes[-5:]) > current_possible_steps)) or ((epoch - first_epoch_with_this_vel) > 100) ) and self.config["speed_factor"] < 0.4:
                        print(" -------------- increase speed --------- ")
                        self.config["speed_factor"] += 0.1
                        self.config["thresh_div"] = 0.1
                        successes = []
                        first_epoch_with_this_vel = epoch + 1
                        self.current_score = 0 if self.suc_up_down == 1 else np.inf

                print(f"\nEpoch {epoch}")
                self.run_epoch(train="controller", epoch=epoch)

                if sampling_based_finetune:
                    print(
                        "Sampled data (exploration):",
                        self.state_data.eval_counter
                    )
                    self.results_dict["samples_in_d2"].append(
                        self.state_data.eval_counter
                    )
                else:
                    print(
                        "Data used for training:",
                        epoch * (self.epoch_size * (1 + self.self_play))
                    )
        except KeyboardInterrupt:
            pass
        # Save model
        self.finalize()

    def run_dynamics(self, config):
        try:
            for epoch in range(config["nr_epochs"]):
                _ = self.evaluate_model(epoch)

                # train dynamics as long as
                # - lower than train_dyn_for_epochs
                # - alternating or use all?
                if (
                    epoch <= config.get("train_dyn_for_epochs", 10)
                    and epoch % config.get("train_dyn_every", 1) == 0
                ):
                    model_to_train = "dynamics"
                else:
                    model_to_train = "controller"

                print(f"\nEpoch {epoch}")
                self.run_epoch(train=model_to_train, epoch=epoch)

                self.results_dict["samples_in_d2"].append(
                    self.count_finetune_data
                )

                if epoch == config["train_dyn_for_epochs"]:
                    print("Params of dynamics model after training:")
                    for key, val in self.train_dynamics.state_dict().items():
                        if len(val) > 10:
                            print(key, "too long")
                            continue
                        print(key, val)
                    self.current_score = 0 if self.suc_up_down == 1 else np.inf

                if epoch <= config.get("train_dyn_for_epochs", 10):
                    print(
                        "Data used to train dynamics:",
                        self.count_finetune_data
                    )

        except KeyboardInterrupt:
            pass
        # Save model
        self.finalize()
