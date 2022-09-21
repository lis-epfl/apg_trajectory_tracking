import torch
import json
import os
from pathlib import Path
import numpy as np
import casadi as ca
import torch.nn as nn
import torch

from neural_control.dynamics.learnt_dynamics import (
    LearntDynamics, LearntDynamicsMPC
)

# target state means that theta is zero --> only third position matters
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.81


class CartpoleDynamics:

    def __init__(self, modified_params={}, test_time=0, batch_size=1):
        self.batch_size = batch_size
        with open(
            os.path.join(
                Path(__file__).parent.absolute(), "config_cartpole.json"
            ), "r"
        ) as infile:
            self.cfg = json.load(infile)

        self.test_time = test_time
        self.cfg.update(modified_params)
        self.cfg["total_mass"] = self.cfg["masspole"] + self.cfg["masscart"]
        self.cfg["polemass_length"] = self.cfg["masspole"] * self.cfg["length"]
        self.timestamp = 0
        # delay of 2
        if self.cfg["delay"] > 0:
            self.action_buffer = np.zeros(
                (batch_size, int(self.cfg["delay"]), 1)
            )
        self.enforce_contact = -1

    def reset_buffer(self):
        self.action_buffer = np.zeros((batch_size, int(self.cfg["delay"]), 1))

    def __call__(self, state, action, dt):
        return self.simulate_cartpole(state, action, dt)

    def simulate_cartpole(self, state, action, dt):
        """
        Compute new state from state and action
        """
        self.timestamp += .05
        # # get action to range [-1, 1]
        # action = torch.sigmoid(action)
        # action = action * 2 - 1
        # # Attempt to use discrete actions
        # if self.test_time:
        #     action = torch.tensor(
        #         [-1]
        #     ) if action[0, 0] > action[0, 1] else torch.tensor([-1])
        # else:
        #     action = torch.softmax(action, dim=1)
        #     action = action[:, 0] * -1 + action[:, 1]
        # get state
        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]
        theta_dot = state[:, 3]
        # (x, x_dot, theta, theta_dot) = state

        # helper variables
        force = self.cfg["max_force_mag"] * action
        # print("actual force", force)
        if self.cfg["delay"] > 0:
            # extract force and update buffer
            force_orig = force.clone()
            force = torch.from_numpy(self.action_buffer[:, -1])
            self.action_buffer = np.roll(self.action_buffer, -1, axis=1)
            self.action_buffer[:, 0] = force_orig.numpy()

        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        sig = self.cfg["muc"] * torch.sign(x_dot)

        # add and multiply
        temp = torch.add(
            torch.squeeze(force),
            self.cfg["polemass_length"] * torch.mul(theta_dot**2, sintheta)
        )
        # divide
        thetaacc = (
            gravity * sintheta - (costheta * (temp - sig)) -
            (self.cfg["mup"] * theta_dot / self.cfg["polemass_length"])
        ) / (
            self.cfg["length"] * (
                4.0 / 3.0 - self.cfg["masspole"] * costheta * costheta /
                self.cfg["total_mass"]
            )
        )
        thetaacc += (1 + np.sin(self.timestamp)) * self.cfg["contact"]
        wind_drag = self.cfg["wind"] * torch.cos(theta * 5)

        # swapped these two lines
        theta = theta + dt * theta_dot
        theta_dot = theta_dot + dt * (thetaacc + wind_drag)

        # add velocity of cart
        xacc = (
            temp - (self.cfg['polemass_length'] * thetaacc * costheta) - sig
        ) / self.cfg["total_mass"]

        # # Contact dynamics on the cart
        # if self.cfg["contact"] > 0 and (
        #     # first possibility: periodically applied
        #     (np.sin(self.timestamp) > 0 and self.enforce_contact == -1)
        #     # second possibility: set manually (for dataset generation!)
        #     or self.enforce_contact == 1
        # ):
        #     xacc += self.cfg["contact"]

        x = x + dt * x_dot
        x_dot = x_dot + dt * (xacc - self.cfg["vel_drag"] * x_dot)

        new_state = torch.stack((x, x_dot, theta, theta_dot), dim=1)
        return new_state


class LearntCartpoleDynamics(LearntDynamics, CartpoleDynamics):

    def __init__(self, modified_params={}, not_trainable=[]):
        CartpoleDynamics.__init__(self, modified_params=modified_params)
        super(LearntCartpoleDynamics, self).__init__(4, 1)

        dict_pytorch = {}
        for key, val in self.cfg.items():
            requires_grad = True
            # # code to avoid training the parameters
            if not_trainable == "all" or key in not_trainable:
                requires_grad = False
            dict_pytorch[key] = torch.nn.Parameter(
                torch.tensor([val]), requires_grad=requires_grad
            )
        self.cfg = torch.nn.ParameterDict(dict_pytorch)

    def simulate(self, state, action, dt):
        return self.simulate_cartpole(state, action, dt)


class SequenceCartpoleDynamics(LearntDynamicsMPC, CartpoleDynamics):

    def __init__(self, buffer_length=3):
        CartpoleDynamics.__init__(self)
        super(SequenceCartpoleDynamics,
              self).__init__(5 * buffer_length, 1, out_state_size=4)

    def simulate(self, state, action, dt):
        return self.simulate_cartpole(state, action, dt)

    def forward(self, state, state_action_buffer, action, dt):
        # run through normal simulator f hat
        new_state = self.simulate(state, action, dt)
        # run through residual network delta
        added_new_state = self.state_transformer(state_action_buffer, action)
        return new_state + added_new_state


class ImageCartpoleDynamics(torch.nn.Module, CartpoleDynamics):

    def __init__(
        self, img_width, img_height, nr_img=5, state_size=4, action_dim=1
    ):
        CartpoleDynamics.__init__(self)
        super(ImageCartpoleDynamics, self).__init__()

        self.img_width = img_width
        self.img_height = img_height
        # conv net
        self.conv1 = nn.Conv2d(nr_img * 2 - 1, 10, 5, padding=2)
        self.conv2 = nn.Conv2d(10, 10, 3, padding=1)
        self.conv3 = nn.Conv2d(10 + 2, 20, 3, padding=1)
        self.conv4 = nn.Conv2d(20, 1, 3, padding=1)

        # residual network
        self.flat_img_size = 10 * (img_width) * (img_height)

        self.linear_act = nn.Linear(action_dim, 32)
        self.act_to_img = nn.Linear(32, img_width * img_height)

        self.linear_state_1 = nn.Linear(self.flat_img_size + 32, 64)
        self.linear_state_2 = nn.Linear(64, state_size, bias=False)

    def conv_head(self, image):
        cat_all = [image]
        for i in range(image.size()[1] - 1):
            cat_all.append(
                torch.unsqueeze(image[:, i + 1] - image[:, i], dim=1)
            )
        sub_images = torch.cat(cat_all, dim=1)
        conv1 = torch.relu(self.conv1(sub_images.float()))
        conv2 = torch.relu(self.conv2(conv1))
        return conv2

    def action_encoding(self, action):
        ff_act = torch.relu(self.linear_act(action))
        return ff_act

    def state_transformer(self, image_conv, act_enc):
        flattened = image_conv.reshape((-1, self.flat_img_size))
        state_action = torch.cat((flattened, act_enc), dim=1)

        ff_1 = torch.relu(self.linear_state_1(state_action))
        ff_2 = self.linear_state_2(ff_1)
        return ff_2

    def image_prediction(self, image_conv, act_enc, prior_img):
        act_img = torch.relu(self.act_to_img(act_enc))
        act_img = act_img.reshape((-1, 1, self.img_width, self.img_height))
        # concat channels
        with_prior = torch.cat((image_conv, prior_img, act_img), dim=1)
        # conv
        conv3 = torch.relu(self.conv3(with_prior))
        conv4 = torch.sigmoid(self.conv4(conv3))
        # return the single channel that we have (instead of squeeze)
        return conv4[:, 0]

    def forward(self, state, image, action, dt):
        # run through normal simulator f hat
        new_state = self.simulate_cartpole(state, action, dt)
        # encode image and action (common head)
        img_conv = self.conv_head(image)
        act_enc = self.action_encoding(action)
        # run through residual network delta
        added_new_state = self.state_transformer(img_conv, act_enc)
        # # Predict next image
        # prior_img = torch.unsqueeze(image[:, 0], 1).float()
        # next_img = self.image_prediction(img_conv, act_enc, prior_img)
        return new_state + added_new_state  # , next_img


class CartpoleDynamicsMPC(CartpoleDynamics):

    def __init__(self, modified_params={}, use_residual=False):
        CartpoleDynamics.__init__(self, modified_params=modified_params)
        self.use_residual = use_residual
        if use_residual and "linear_state_1.weight" in modified_params:
            print("Set weights for residual in MPC F")
            self.weight1 = modified_params["linear_state_1.weight"]
            # self.bias1 = modified_params["linear_state_1.bias"]
            self.weight2 = modified_params["linear_state_2.weight"]
            self.weight3 = modified_params["linear_state_3.weight"]
        elif len(modified_params) > 0:
            print("Using identified system but only parameters, no res")
            self.use_residual = False

    def simulate_cartpole(self, dt):
        (x, x_dot, theta, theta_dot) = (
            ca.SX.sym("x"), ca.SX.sym("x_dot"), ca.SX.sym("theta"),
            ca.SX.sym("theta_dot")
        )
        action = ca.SX.sym("action")
        # x_state = ca.vertcat(x, x_dot, theta, theta_dot)

        # first part is state
        current_state = ca.vertcat(x, x_dot, theta, theta_dot)

        # rest of history are states and actions beforehand
        x_h4 = ca.SX.sym('h4')
        x_h5 = ca.SX.sym('h5')
        x_h6 = ca.SX.sym('h6')
        x_h7 = ca.SX.sym('h7')
        x_h8 = ca.SX.sym('h8')
        x_h9 = ca.SX.sym('h9')
        x_h10 = ca.SX.sym('h10')
        x_h11 = ca.SX.sym('h11')
        x_h12 = ca.SX.sym('h12')
        x_h13 = ca.SX.sym('h13')
        x_h14 = ca.SX.sym('h14')

        x_state = ca.vertcat(
            current_state, x_h4, x_h5, x_h6, x_h7, x_h8, x_h9, x_h10, x_h11,
            x_h12, x_h13, x_h14
        )

        # helper variables
        force = self.cfg["max_force_mag"] * action
        costheta = ca.cos(theta)
        sintheta = ca.sin(theta)
        sig = self.cfg["muc"] * ca.sign(x_dot)

        # add and multiply
        temp = force + self.cfg["polemass_length"] * theta_dot**2 * sintheta

        # divide
        thetaacc = (
            gravity * sintheta - (costheta * (temp - sig)) -
            (self.cfg["mup"] * theta_dot / self.cfg["polemass_length"])
        ) / (
            self.cfg["length"] * (
                4.0 / 3.0 - self.cfg["masspole"] * costheta * costheta /
                self.cfg["total_mass"]
            )
        )
        wind_drag = self.cfg["wind"] * costheta

        # add velocity of cart
        x_acc = (
            temp - (self.cfg['polemass_length'] * thetaacc * costheta) - sig
        ) / self.cfg["total_mass"]

        x_state_dot = ca.vertcat(x_dot, x_acc, theta_dot, thetaacc + wind_drag)

        if self.use_residual:
            print("USING res in mpc function")
            history = ca.vertcat(x_state, action)
            # state_action = ca.vertcat(
            #     x_state, action, x_state, action, x_state, action, action
            # )
            # residual_state_1 = ca.tanh(self.weight1 @ history + self.bias1)
            # residual_state = self.weight2 @ residual_state_1
            residual_state_1 = ca.tanh(self.weight1 @ history)
            residual_state_2 = ca.tanh(self.weight2 @ residual_state_1)
            residual_state = ca.tanh(self.weight3 @ residual_state_2)
        else:
            residual_state = 0

        new_x_state = current_state + dt * x_state_dot + residual_state
        X = ca.vertcat(
            new_x_state, action, current_state, x_h4, x_h5, x_h6, x_h7, x_h8,
            x_h9
        )

        F = ca.Function('F', [x_state, action], [X], ['x', 'u'], ['ode'])
        return F


if __name__ == "__main__":
    state_test_np = np.array([0.5, 1.3, 0.1, 0.4])
    state_test = torch.unsqueeze(torch.from_numpy(state_test_np), 0).float()
    action_test_np = np.array([0.4])
    action_test = torch.unsqueeze(torch.from_numpy(action_test_np), 0).float()

    normal_dyn = CartpoleDynamics()
    next_state = normal_dyn(state_test, action_test, 0.02)
    print("------------")
    print(next_state[0])

    # test: compare to mpc
    # if test doesnt work, remove clamp!!
    mpc_dyn = CartpoleDynamicsMPC()
    F = mpc_dyn.simulate_cartpole(0.02)
    mpc_state = F(state_test_np, action_test_np)
    print("--------------------")
    print(mpc_state)
