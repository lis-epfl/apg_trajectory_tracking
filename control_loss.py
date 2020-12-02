import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# target state means that theta is zero --> only third position matters
# TODO: can check at a later stage whether it makes sense to also have theta
# dot equal to zero (rather no velocity)
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
max_force_mag = 30.0
tau = 0.02  # seconds between state updates
muc = 0.0005
mup = 0.000002


def state_to_theta(x_dot, theta, theta_dot, action):
    # compute next state
    force = max_force_mag * action
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    sig = muc * torch.sign(x_dot)
    # add and multiply
    temp = torch.add(
        torch.squeeze(force),
        polemass_length * torch.mul(theta_dot**2, sintheta)
    )
    # divide
    thetaacc = (
        gravity * sintheta - (costheta * (temp - sig)) -
        (mup * theta_dot / polemass_length)
    ) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))

    # swapped these two lines
    theta_dot = theta_dot + tau * thetaacc
    theta = theta + tau * theta_dot

    # add velocity of cart
    xacc = (temp - (polemass_length * thetaacc * costheta) - sig) / total_mass
    x_dot = x_dot + tau * xacc

    return x_dot, theta, theta_dot


def control_loss_function(action, state, lambda_factor=.4, printout=0):

    # bring action into -1 1 range
    action = torch.sigmoid(action) - .5
    # get state
    x_dot_orig = state[:, 1]
    theta_orig = state[:, 2]
    theta_dot_orig = state[:, 3]
    nr_actions = action.size()[1]
    # normalize
    # theta_normed = theta.clone()
    # torch.sign(theta) * torch.maximum(
    #     torch.abs(theta),
    #     torch.ones(theta.size()) * .1
    # )

    # check the maximum possible force we can apply
    direction = torch.sign(theta_orig)
    action_opp_direction = direction * torch.ones(x_dot_orig.size()) * .5 * .5
    x_dot_max_orig, theta_max_orig, theta_dot_max_orig = (
        x_dot_orig, theta_orig, theta_dot_orig
    )

    # set current state
    x_dot, theta, theta_dot = (x_dot_orig, theta_orig, theta_dot_orig)
    # print("theta_orig", theta_orig)
    # Run in loop
    for i in range(nr_actions):
        x_dot, theta, theta_dot = state_to_theta(
            x_dot, theta, theta_dot, action[:, i]
        )
        # print("action", action[:, i])
        # print("theta in loop", theta)

        # # execute with the maximum force
        # print(action_opp_direction[0].item())
        # print("current best theta", theta_max_orig[0].item())
        x_dot_max_orig, theta_max_orig, theta_dot_max_orig = state_to_theta(
            x_dot_max_orig, theta_max_orig, theta_dot_max_orig,
            action_opp_direction
        )

    # maximum possible musn't cross the zero point
    theta_max_possible = torch.maximum(
        theta_max_orig * direction, torch.zeros(theta_orig.size())
    ) * direction

    # Compute loss: normalized version:
    # # Working version:
    # print("theta", theta)
    # print("theta_max_possible", theta_max_possible)
    angle_loss = (theta - theta_max_possible)**2

    # angle_acc = torch.abs(theta_dot) - torch.abs(theta_dot_orig)
    cart_acc = torch.abs(x_dot) - torch.abs(x_dot_orig)
    # # loss = angle_loss + velocity_loss

    # # New version for swing up --> minimize angle acceleration in upper part
    # # and maximize it in lower part
    factor = 2 * torch.cos(theta) + 1  # shift up: 2 * torch.cos(theta) + 1
    # # norm on action to prohibit large push the whole time plus
    # # angle loss
    # loss = .2 * (1 + factor) * angle_loss + factor * (angle_acc + cart_acc)
    loss = angle_loss  # + factor * cart_acc
    # print("orig:")
    if printout:
        print("actions:", action[0])
        print(
            "theta before", theta_orig[0].item(), "theta after",
            theta[0].item(), "theta max possible", theta_max_possible[0].item()
        )
        # print("action", action)
        # print("factor", factor)
        # print("prev theta dot:", theta_dot_orig)
        # print("now theta dot:", theta_dot)
        # print()
        # print("cart acc loss", cart_acc)
        # print("losses:")
        print("angle loss", angle_loss[0].item())
        print("factor:", factor)
        print("angle_acc", angle_acc)
        # print("angle acc loss", angle_acc)
    # print(fail)
    return torch.sum(loss)  # + angle_acc)
