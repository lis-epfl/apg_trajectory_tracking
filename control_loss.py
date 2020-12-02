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


def state_to_theta(state, action):
    """
    Compute new state from state and action
    """
    # get state
    (x, x_dot, theta, theta_dot) = state

    # helper variables
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
    x = x + tau * x_dot

    return (x, x_dot, theta, theta_dot)


def control_loss_function(action, state, lambda_factor=.4, printout=0):

    # bring action into -1 1 range
    action = torch.sigmoid(action) - .5
    # get state
    x_orig = state[:, 0]
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
    state = (x_orig, x_dot_orig, theta_orig, theta_dot_orig)

    # check the maximum possible force we can apply
    direction = torch.sign(theta_orig)
    action_opp_direction = direction * torch.ones(x_dot_orig.size()) * .5 * .5
    state_max = state

    # check which direction for position --> if we are in positive part, we
    # need to move left (negative action)
    # direction = torch.sign(x_orig)
    # action_opp_position = direction * (-1) * torch.ones(
    #     x_dot_orig.size()
    # ) * .5 * .5
    # state_pos = state

    for i in range(nr_actions):
        state = state_to_theta(state, action[:, i])

        # execute with the maximum force
        state_max = state_to_theta(state_max, action_opp_direction)

        # execute to get the cart in the middle
        # state_pos = state_to_theta(state_pos, action_opp_position)

    # extract necessary variables from state
    (x, x_dot, theta, theta_dot) = state

    # maximum possible musn't cross the zero point
    theta_max_possible = torch.maximum(
        state_max[2] * direction, torch.zeros(theta_orig.size())
    ) * direction

    # # same for position
    # pos_max_possible = torch.maximum(
    #     state_pos[0] * direction, torch.zeros(theta_orig.size())
    # ) * direction

    # Compute loss: normalized version:
    angle_loss = (theta - theta_max_possible)**2
    position_loss = (x)**2

    # print("x_dot", x_dot_orig[0].item())
    # print("x orig", x_orig[0].item())
    # print("x", x[0].item())
    # print("max pos", pos_max_possible[0].item())
    # print(position_loss[0].item())
    # print()

    # angle_acc = torch.abs(theta_dot) - torch.abs(theta_dot_orig)
    # cart_acc = torch.abs(x_dot) - torch.abs(x_dot_orig)

    # Cosine loss: minimize angle acceleration in upper part
    # and maximize it in lower part
    # factor = 2 * torch.cos(theta) + 1  # shift up: 2 * torch.cos(theta) + 1

    # loss = .2 * (1 + factor) * angle_loss + factor * (angle_acc + cart_acc)
    # the higher the angle in the beginning, the more we allow to move the cart
    regularize_x = torch.maximum(
        torch.zeros(theta_orig.size()), (3 - np.abs(theta_orig))
    )
    loss = angle_loss + lambda_factor * 0.05 * regularize_x * position_loss

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
        print("position_loss", position_loss[0].item())
        print(
            "position_loss actual",
            (.05 * lambda_factor * regularize_x * position_loss)[0].item()
        )
        # print("x", x[0].item())
        # print("x actual", (.1 * lambda_factor * regularize_x * x**2)[0].item())
        print("angle loss", angle_loss[0].item())
        # print("angle acc loss", angle_acc)
    # print(fail)
    return torch.sum(loss)  # + angle_acc)
