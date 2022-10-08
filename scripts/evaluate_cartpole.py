import os
import time
import argparse
import json
import numpy as np
import torch
try:
    import cv2
    INSTALLED_CV2 = True
except ImportError:
    INSTALLED_CV2 = False
    print(
        "Warning: cv2 not installed, must be installed for cartpole image experiments"
    )

from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, SequenceCartpoleDynamics
)
from neural_control.controllers.mpc import MPC
from neural_control.controllers.network_wrapper import (
    CartpoleWrapper, CartpoleImageWrapper, SequenceCartpoleWrapper
)
from neural_control.models.simple_model import Net, ImageControllerNet

# mpc like receding horizon
APPLY_UNTIL = 1

# current state, predicted action, images of prev states, next state after act
collect_states, collect_actions, collect_img = [], [], []
buffer_len = 4
img_width, img_height = (200, 300)
crop_width = 60
center_at_x = True


class Evaluator:

    def __init__(
        self, controller, eval_env, eval_dyn=None, collect_image_dataset=0
    ):
        self.controller = controller
        self.eval_env = eval_env
        self.eval_dyn = eval_dyn
        self.mpc = isinstance(self.controller, MPC)
        self.image_dataset = collect_image_dataset
        self.init_buffers()
        self.initialize_straight = 1

    def init_buffers(self):
        self.image_buffer = np.zeros((buffer_len, img_width, img_height))
        self.state_buffer = np.zeros((buffer_len, 4))
        self.action_buffer = np.zeros((buffer_len, 1))

    def _preprocess_img(self, image):
        resized = cv2.resize(
            np.mean(image, axis=2),
            dsize=(img_height, img_width),
            interpolation=cv2.INTER_LINEAR
        )
        return ((255 - resized) > 0).astype(float)

    def _convert_image_buffer(self, state, crop_width=crop_width):
        # image and corresponding state --> normalize x pos in image buffer!
        img_width_half = self.image_buffer.shape[2] // 2
        if center_at_x:
            x_pos = state[0] / self.eval_env.state_limits[0]
            x_img = int(img_width_half + x_pos * img_width_half)
            use_img_buffer = np.roll(
                self.image_buffer.copy(), img_width_half - x_img, axis=2
            )
            return use_img_buffer[:, 75:175, img_width_half -
                                  crop_width:img_width_half + crop_width]
        else:
            x_img = img_width_half
            return self.image_buffer[:, 75:175,
                                     x_img - crop_width:x_img + crop_width]

    def evaluate_in_environment(
        self,
        nr_iters=1,
        max_steps=250,
        render=False,
        burn_in_steps=50,
        return_success=0
    ):
        """
        Measure success --> how long can we balance the pole on top
        """
        self.dyn_eval_test = []
        if nr_iters == 0:
            return 0, 0, []
        data_collection = []
        velocities = []
        with torch.no_grad():
            success = np.zeros(nr_iters)
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for n in range(nr_iters):
                # only set the theta to the top, and reduce speed
                self.eval_env._reset_upright()
                # TEST IN SAME DISTRIBUTION
                if self.initialize_straight:
                    # self.eval_env.state = (np.random.rand(4) - .5) * .1
                    # x normal distributed --> if centered, then doesn't matter
                    if not center_at_x:
                        self.eval_env.state[0] = np.random.randn() / 2.5
                    else:
                        self.eval_env.state[0] = 0
                    # small velocity
                    self.eval_env.state[1] = 0  # (np.random.rand() - .5) * .3
                    # zero angle! because otherwise hardly collecting data
                    self.eval_env.state[2] = 0  # (np.random.rand() - .5) * .1
                    self.eval_env.state[3] = 0

                new_state = self.eval_env.state
                if render and INSTALLED_CV2:
                    start_img = self._preprocess_img(
                        self.eval_env._render(mode="rgb_array")
                    )
                    self.image_buffer = np.array(
                        [start_img for _ in range(buffer_len)]
                    )

                angles = list()
                # Start balancing
                for i in range(max_steps):
                    # Transform state in the same way as the training data
                    # and normalize
                    # Prepare img seq
                    converted_img_seq = torch.from_numpy(
                        np.expand_dims(
                            self._convert_image_buffer(new_state)[:-1], 0
                        )
                    ).float()
                    # prepare state action history
                    state_buffer = torch.from_numpy(
                        np.expand_dims(self.state_buffer.copy()[:-1], 0)
                    ).float()
                    action_buffer = torch.from_numpy(
                        np.expand_dims(self.action_buffer.copy()[:-1], 0)
                    ).float()
                    state_action_history = torch.cat(
                        (state_buffer, action_buffer), dim=2
                    )
                    network_input = torch.reshape(
                        state_action_history, (
                            -1, state_action_history.size()[1] *
                            state_action_history.size()[2]
                        )
                    )
                    # ------------- Predict action ------------------
                    if self.image_dataset:
                        action_seq = torch.rand(1, 4) - .5
                    else:
                        if isinstance(
                            self.controller, SequenceCartpoleWrapper
                        ):
                            action_seq = self.controller.predict_actions(
                                state_buffer, action_buffer, network_input
                            )
                        elif isinstance(self.controller, CartpoleImageWrapper):
                            action_seq = self.controller.predict_actions(
                                converted_img_seq,
                                self.state_buffer.copy()[:-1]
                            )
                        else:
                            action_seq = self.controller.predict_actions(
                                new_state, network_input
                            )
                            if self.mpc:
                                action_seq = torch.tensor([action_seq])

                    if i % 10 == 0 and self.eval_dyn is not None:
                        self.dyn_eval_test.append(
                            dyn_comparison_cartpole(
                                self.eval_dyn,
                                new_state,
                                action_seq[0, 0],
                                network_input,
                                self.eval_env.dynamics.timestamp,
                                dt=self.eval_env.dt
                            )
                        )
                    # ------------- Take step with dynamics ------------------
                    prev_state = new_state.copy()
                    for action_ind in range(APPLY_UNTIL):
                        # run action in environment
                        new_state = self.eval_env._step(
                            action_seq[:, action_ind],
                            image=converted_img_seq,
                            state_action_buffer=network_input,
                            is_torch=True
                        )
                        data_collection.append(new_state)
                        velocities.append(float(np.absolute(new_state[1])))
                        if i > burn_in_steps:
                            angles.append(np.absolute(new_state[2]))
                        if render:
                            new_img = self.eval_env._render(mode="rgb_array")
                            # test = self.eval_env._render(mode="rgb_array")
                            # time.sleep(.1)

                    # update image buffer with new image
                    if render and INSTALLED_CV2:
                        self.image_buffer = np.roll(
                            self.image_buffer, 1, axis=0
                        )
                        self.image_buffer[0] = self._preprocess_img(new_img)

                    # update state buffer with new state
                    self.state_buffer = np.roll(self.state_buffer, 1, axis=0)
                    self.state_buffer[0] = new_state
                    self.action_buffer = np.roll(self.action_buffer, 1, axis=0)
                    self.action_buffer[0] = action_seq[0, 0].numpy()

                    # save for image task
                    if self.image_dataset and i > buffer_len:
                        assert APPLY_UNTIL == 1
                        collect_states.append(self.state_buffer.copy())
                        collect_actions.append(self.action_buffer.copy())
                        collect_img.append(
                            self._convert_image_buffer(prev_state)
                        )

                    if not self.eval_env.is_upright():
                        break
                        # track number of timesteps until failure

                avg_angle[n] = np.mean(angles) if len(angles) > 0 else 100
                success[n] = i
                self.eval_env._reset()
        res = {
            "mean_vel": np.mean(velocities),
            "std_vel": np.std(velocities),
            "mean_stable": np.mean(success),
            "std_stable": np.std(success)
        }
        dyn_eval_test = np.array(self.dyn_eval_test)
        if len(self.dyn_eval_test) > 0:
            print(
                "Dynamic gap: %3.2f (%3.2f)" %
                (np.mean(dyn_eval_test[:, 0]), np.std(dyn_eval_test[:, 0]))
            )
            print(
                "Dynamic trained: %3.2f (%3.2f)" %
                (np.mean(dyn_eval_test[:, 1]), np.std(dyn_eval_test[:, 1]))
            )
            res["mean_dyn_trained"] = np.mean(dyn_eval_test[:, 1])
            res["std_dyn_trained"] = np.std(dyn_eval_test[:, 1])
            res["mean_dyn_gap"] = np.mean(dyn_eval_test[:, 0])
        mean_err = np.mean(success)
        std_err = np.std(success)
        if not self.image_dataset:
            print(
                "Average velocity: %3.2f (%3.2f)" %
                (np.mean(velocities), np.std(velocities))
            )
            print("Average success: %3.2f (%3.2f)" % (mean_err, std_err))
        if return_success:
            return success, velocities
        return res

    def evaluate_swingup(
        self,
        nr_iters=1,
        max_steps=250,
        render=False,
        burn_in_steps=100,
        return_success=0
    ):
        avg_state = []
        velocities = []
        with torch.no_grad():
            success = np.zeros(nr_iters)
            # observe also the oscillation
            avg_angle = np.zeros(nr_iters)
            for n in range(nr_iters):
                # initialize success (upright in intermediate checks) to true
                is_upright = True
                self.eval_env._reset_swingup()
                for i in range(max_steps):
                    new_state = self.eval_env.state
                    # print(new_state)
                    action_seq = self.controller.predict_actions(
                        new_state, new_state
                    )
                    # print(action_seq[:, 0])

                    # run action in environment
                    new_state = self.eval_env._step(
                        action_seq[:, 0],
                        image=self.image_buffer,
                        state_action_buffer=new_state,
                        is_torch=True
                    )
                    if i > burn_in_steps:
                        velocities.append(new_state[1])
                        avg_state.append(new_state.copy())
                        # set upright to false as soon as it was one time lower
                        if new_state[2] > 1:
                            is_upright = False
                    if render:
                        self.eval_env._render()
                        time.sleep(0.05)
                success[n] = int(is_upright)

        avg_state = np.sqrt(np.array(avg_state)**2)
        mean_state = np.mean(avg_state, axis=0)
        std_state = np.std(avg_state, axis=0)
        print("average states", [round(e, 2) for e in mean_state])
        # print("std", std_state)
        res_eval = {}
        res_eval["mean_vel"] = float(np.mean(np.absolute(velocities)))
        res_eval["std_vel"] = float(np.mean(np.absolute(velocities)))
        if return_success:
            return success
        return res_eval


def run_saved_arr(path):
    """
    Load a saved sequence of states and visualize it
    """
    states = np.load(path)
    self.eval_env = CartPoleEnv()
    for state in states:
        self.eval_env.state = state
        self.eval_env._render()
        time.sleep(.1)


def load_model(model_name, epoch, is_seq=False):
    with open(
        os.path.join("trained_models", "cartpole", model_name, "config.json"),
        "r"
    ) as infile:
        config = json.load(infile)

    path_load = os.path.join(
        "trained_models", "cartpole", model_name, "model_pendulum" + epoch
    )
    if not os.path.exists(path_load):
        path_load = os.path.join(
            "trained_models", "cartpole", model_name, "model_cartpole" + epoch
        )
    net = torch.load(path_load)
    # some_dataset = CartpoleSequenceDataset(
    #     load_data_path="data/cartpole_img_5.npz", use_samples=2
    # )
    # not necessary for image eval because dataset not used for preprocessing
    # some_dataset = CartpoleImageDataset(
    #     load_data_path="data/cartpole_img_16.npz"
    # )
    config["self_play"] = 0
    net.eval()
    if isinstance(net, Net):
        if is_seq:
            controller_model = SequenceCartpoleWrapper(
                net, some_dataset, **config
            )
        else:
            controller_model = CartpoleWrapper(net, **config)
    else:
        controller_model = CartpoleImageWrapper(net, some_dataset, **config)
    return controller_model


if __name__ == "__main__":
    # make as args:
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="current_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-e", "--epoch", type=str, default="", help="Saved epoch"
    )
    parser.add_argument(
        "-a", "--eval", type=int, default=0, help="number eval runs"
    )
    parser.add_argument(
        "-save_data",
        action="store_true",
        help="save the episode as training data"
    )
    parser.add_argument(
        "-d", "--dataset", type=int, default=0, help="number collect dataset"
    )
    args = parser.parse_args()

    # PARAMs
    dt = 0.05
    thresh_div = 0.3

    if args.model == "mpc":
        load_dynamics = None
        # "trained_models/cartpole/pretrained_sampling_for_rl_comp_plot/con_seq_500/dynamics_model"
        controller_model = MPC(
            horizon=10,
            dt=dt,
            dynamics="cartpole",
            load_dynamics=load_dynamics
        )
    else:
        controller_model = load_model(
            args.model,
            args.epoch,
            is_seq=("seq" in args.model or "contact" in args.model)
        )

    modified_params = {}  # {"contact": 1}
    # {"wind": .5}
    # wind 0.01 works for wind added to x directlt, needs much higher (.5)
    # to affect the acceleration much

    # define dynamics and environmen
    dynamics = CartpoleDynamics(modified_params=modified_params)
    eval_env = CartPoleEnv(dynamics, dt, thresh_div=thresh_div)

    dyn_trained = None
    try:
        dyn_trained = SequenceCartpoleDynamics(buffer_length=3)
        dyn_trained.load_state_dict(
            torch.load(
                os.path.join(
                    "trained_models/cartpole/" + args.model, "dynamics_model"
                )
            ),
            strict=False
        )
        print("Loaded dynamics model from ")
    except:
        dyn_trained = None
        print("in this saved model no dynamics model is found")

    evaluator = Evaluator(controller_model, eval_env, eval_dyn=dyn_trained)
    # angles = evaluator.run_for_fixed_length(net, render=True)

    if args.dataset > 0:
        evaluator.image_dataset = 1
        counter = 0
        while len(collect_actions) < args.dataset:
            counter += 1
            evaluator.init_buffers()
            _ = evaluator.evaluate_in_environment(render=True, max_steps=400)
        collect_actions = np.array(collect_actions)
        # cut off bottom and top
        collect_img = np.array(collect_img)
        collect_states = np.array(collect_states)
        print(collect_states.shape, collect_actions.shape, collect_img.shape)
        np.savez(
            f"data/cartpole_img_{args.dataset}.npz", collect_img,
            collect_actions, collect_states
        )
    elif args.eval > 0:
        # set to random initial state
        evaluator.initialize_straight = False
        res_eval = evaluator.evaluate_in_environment(
            render=False,
            max_steps=250,
            nr_iters=args.eval,
            return_success=False
        )
        with open(
            f"../presentations/final_res/cartpole_seq_plot/{args.model}.json",
            "w"
        ) as outfile:
            json.dump(res_eval, outfile)
        # np.savez(
        #     f"../presentations/final_res/cartpole_seq_plot/{args.model}.npz",
        #     np.array(successes), np.array(velocities)
        # )
    else:
        evaluator.initialize_straight = False
        _ = evaluator.evaluate_swingup(render=True, max_steps=500, nr_iters=1)
