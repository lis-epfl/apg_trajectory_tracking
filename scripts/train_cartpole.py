import os
import argparse
import numpy as np
import json
import torch
import torch.optim as optim

from train_base import TrainBase
from neural_control.dataset import (CartpoleDataset)
from neural_control.drone_loss import (
    cartpole_loss_balance, cartpole_loss_swingup, cartpole_loss_mpc
)
from evaluate_cartpole import Evaluator
from neural_control.models.simple_model import (
    Net, ImageControllerNet, ImageControllerNetDQN, StateToImg
)
from neural_control.plotting import plot_loss, plot_success
from neural_control.environments.cartpole_env import (
    construct_states, CartPoleEnv
)
from neural_control.controllers.network_wrapper import (
    CartpoleWrapper, CartpoleImageWrapper, SequenceCartpoleWrapper
)
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, LearntCartpoleDynamics, ImageCartpoleDynamics,
    SequenceCartpoleDynamics
)


class TrainCartpole(TrainBase):
    """
    Train a controller for a quadrotor
    """

    def __init__(
        self,
        train_dynamics,
        eval_dynamics,
        config,
        train_image_dyn=0,
        train_seq_dyn=0,
        swingup=0
    ):
        """
        param sample_in: one of "train_env", "eval_env"
        """
        self.swingup = swingup
        self.config = config
        super().__init__(train_dynamics, eval_dynamics, **self.config)
        if self.sample_in == "eval_env":
            self.eval_env = CartPoleEnv(self.eval_dynamics, self.delta_t)
        elif self.sample_in == "train_env":
            self.eval_env = CartPoleEnv(self.train_dynamics, self.delta_t)
        else:
            raise ValueError("sample in must be one of eval_env, train_env")

        # for image processing
        self.train_image_dyn = train_image_dyn
        self.train_seq_dyn = train_seq_dyn

        # state to image transformer:
        self.state_to_img_net = None
        # self.state_to_img_net = torch.load(
        #     "trained_models/cartpole/state_img_net"
        # )
        # self.state_to_img_optim = optim.SGD(
        #     self.state_to_img_net.parameters(), lr=0.0001, momentum=0.9
        # )

    def initialize_model(
        self,
        base_model=None,
        base_model_name="model_cartpole",
        load_dataset="data/cartpole_img_20.npz",
        load_state_to_img=None
    ):
        if base_model is not None:
            self.net = torch.load(os.path.join(base_model, base_model_name))
        else:
            self.net = Net(self.state_size, self.nr_actions * self.action_dim)
        self.state_data = CartpoleDataset(
            num_states=self.config["sample_data"], **self.config
        )
        self.model_wrapped = CartpoleWrapper(self.net, **self.config)

        with open(os.path.join(self.save_path, "config.json"), "w") as outfile:
            json.dump(self.config, outfile)

        # load state to img netwrok if it was finetuned
        if load_state_to_img is not None:
            self.state_to_img_net.load_state_dict(
                torch.load(os.path.join(load_state_to_img, "state_to_img"))
            )

        self.init_optimizer()
        self.config["thresh_div"] = self.config["thresh_div_start"]

    def make_reference(self, current_state):
        ref_states = torch.zeros(
            current_state.size()[0], self.nr_actions, self.state_size
        )
        for k in range(self.nr_actions - 1):
            ref_states[:, k] = (
                current_state * (1 - 1 / (self.nr_actions - 1) * k)
            )
        return ref_states

    def loss_logging(self, epoch_loss, train="controller"):
        self.results_dict["loss_" + train].append(epoch_loss)
        # if train == "controller" or self.epoch % 10 == 0:
        print(f"Loss ({train}): {round(epoch_loss, 2)}")
        # self.writer.add_scalar("Loss/train", epoch_loss)

    def run_epoch(self, train="controller"):
        self.results_dict["trained"].append(train)
        # tic_epoch = time.time()
        running_loss = 0
        for i, data in enumerate(self.trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, current_state = data

            actions = self.net(in_state)
            action_seq = torch.reshape(
                actions, (-1, self.nr_actions, self.action_dim)
            )

            if train == "controller":
                # zero the parameter gradients
                self.optimizer_controller.zero_grad()
                ref_states = self.make_reference(current_state)

                intermediate_states = torch.zeros(
                    current_state.size()[0], self.nr_actions, self.state_size
                )
                for k in range(action_seq.size()[1]):
                    current_state = self.train_dynamics(
                        current_state, action_seq[:, k], dt=self.delta_t
                    )
                    intermediate_states[:, k] = current_state
                # Loss
                loss = cartpole_loss_mpc(
                    intermediate_states, ref_states, action_seq
                )

                loss.backward()
                # for name, param in self.net.named_parameters():
                #     if param.grad is not None:
                #         self.writer.add_histogram(name + ".grad", param.grad)
                #         self.writer.add_histogram(name, param)
                self.optimizer_controller.step()
            else:
                # should work for both recurrent and normal
                loss = self.train_dynamics_model(current_state, action_seq)
                self.count_finetune_data += len(current_state)

            running_loss += loss.item()
        # time_epoch = time.time() - tic
        epoch_loss = running_loss / i
        self.loss_logging(epoch_loss, train=train)
        return epoch_loss

    def evaluate_model(self, epoch):

        eval_dyn = self.train_dynamics if isinstance(
            self.train_dynamics, SequenceCartpoleDynamics
        ) else None

        self.model_wrapped.self_play = self.config["self_play"]
        evaluator = Evaluator(
            self.model_wrapped, self.eval_env, eval_dyn=eval_dyn
        )
        # Start in upright position and see how long it is balaned
        if self.swingup:
            res_eval = evaluator.evaluate_swingup(
                nr_iters=10, render=self.train_image_dyn
            )
        else:
            res_eval = evaluator.evaluate_in_environment(
                nr_iters=10, render=self.train_image_dyn
            )
        success_mean = res_eval["mean_vel"]
        success_std = res_eval["std_vel"]
        for key, val in res_eval.items():
            self.results_dict[key].append(val)
        self.results_dict["evaluate_at"].append(epoch)
        self.save_model(epoch, success_mean, success_std)

        # increase thresholds
        if epoch % 3 == 0 and self.config["thresh_div"] < self.thresh_div_end:
            self.config["thresh_div"] += self.config["thresh_div_step"]
            print(
                "Curriculum learning: increase divergence threshold to",
                self.config["thresh_div"]
            )

        if (epoch + 1) % self.config["resample_every"] == 0:
            print("resample data...")
            self.state_data.resample_data(
                self.config["sample_data"], self.config["thresh_div"]
            )

    def finalize(self):
        torch.save(
            self.state_to_img_net.state_dict(),
            os.path.join(self.save_path, "state_img_net")
        )
        super().finalize(plot_loss="loss_dynamics")

    def collect_data(self):
        self.state_data.resample_data(
            self.config["sample_data"], self.config["thresh_div"]
        )


def train_control(base_model, config, swingup=0):
    """
    Train a controller from scratch or with an initial model
    """
    config["learning_rate_controller"] = 1e-5
    # modified params (usually empty dict)
    modified_params = config["modified_params"]
    train_dynamics = CartpoleDynamics(modified_params)
    eval_dynamics = CartpoleDynamics(modified_params, test_time=1)

    trainer = TrainCartpole(
        train_dynamics, eval_dynamics, config, swingup=swingup
    )
    trainer.initialize_model(base_model)
    try:
        for epoch in range(trainer.config["nr_epochs"]):
            trainer.evaluate_model(epoch)
            print()
            print("Epoch", epoch)
            trainer.run_epoch(train="controller")
    except KeyboardInterrupt:
        pass
    trainer.finalize()


def train_norm_dynamics(base_model, config, not_trainable="all"):
    """First train dynamcs, then train controller with estimated dynamics

    Args:
        base_model (filepath): Model to start training with
        config (dict): config parameters
    """
    modified_params = config["modified_params"]
    config["sample_in"] = "train_env"
    config["thresh_div_start"] = 0.2
    config["train_dyn_every"] = 1

    # train environment is learnt
    train_dyn = LearntCartpoleDynamics(not_trainable=not_trainable)
    eval_dyn = CartpoleDynamics(modified_params=modified_params)
    trainer = TrainCartpole(train_dyn, eval_dyn, config)
    trainer.initialize_model(base_model)
    # RUN
    trainer.run_dynamics(config)


if __name__ == "__main__":
    # LOAD CONFIG - select balance or swigup
    with open("configs/cartpole_config.json", "r") as infile:
        config = json.load(infile)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--todo",
        type=str,
        default="pretrain",
        help="what to do - pretrain, adapt or finetune"
    )
    parser.add_argument(
        "-m",
        "--model_load",
        type=str,
        default="trained_models/cartpole/current_model",
        help="Model to start with (default: None - from scratch)"
    )
    parser.add_argument(
        "-s",
        "--save_name",
        type=str,
        default="test",
        help="Name under which the trained model shall be saved"
    )
    parser.add_argument(
        "-p",
        "--params_trainable",
        type=bool,
        default=False,
        help="Train the parameters of \hat{f} (1) or only residual (0)"
    )
    args = parser.parse_args()

    baseline_model = args.model_load
    baseline_dyn = None
    trainable_params = args.params_trainable
    config["save_name"] = args.save_name

    if args.todo == "pretrain":
        # No baseline model used
        train_control(None, config, swingup=1)
    elif args.todo == "adapt":
        mod_params = {"wind": 0.5}
        config["modified_params"] = mod_params
        trainable_params = [] if args.params_trainable else "all"
        print(
            f"start from pretrained model {args.model_load}, consider scenario\
                {mod_params}, train also parameters - {trainable_params}\
                save adapted dynamics and controller at {args.save_name}"
        )
        # Run
        train_norm_dynamics(
            baseline_model, config, not_trainable=trainable_params
        )
