import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# target state means that theta is zero --> only third position matters
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
max_force_mag = 40.0
tau = 0.02  # seconds between state updates
muc = 0.0005
mup = 0.000002


def state_to_theta(state, action):
    """
    Compute new state from state and action
    """
    # get state
    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    theta_dot = state[:, 3]
    # (x, x_dot, theta, theta_dot) = state

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
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    # add velocity of cart
    xacc = (temp - (polemass_length * thetaacc * costheta) - sig) / total_mass
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc

    new_state = torch.stack((x, x_dot, theta, theta_dot), dim=1)
    return new_state


def control_loss_function(action, state, lambda_factor=.4, printout=0):

    # bring action into -1 1 range
    action = torch.sigmoid(action) - .5

    nr_actions = action.size()[1]

    # update state iteratively for each proposed action
    for i in range(nr_actions):
        state = state_to_theta(state, action[:, i])
    abs_state = torch.abs(state)

    pos_loss = state[:, 0]**2
    # velocity losss is low when x is high
    vel_loss = .1 * abs_state[:, 1] * (2.4 - abs_state[:, 0])**2
    angle_loss = abs_state[:, 2] + .1 * abs_state[:, 3]
    loss = .1 * (pos_loss + vel_loss + 2 * angle_loss)
    # .1 * torch.mv(abs_state, weighting)  # * prev_weighted
    # print(state**2)
    # execute with the maximum force
    # state_max = state_to_theta(state_max, action_opp_direction)

    # loss += (state[2] - state_max[2])**2
    # print(loss)
    # execute to get the cart in the middle
    # state_pos = state_to_theta(state_pos, action_opp_position)
    # loss += 5 * (state[:, 0] - state_pos[:, 0])**2

    # loss += .1 * state[0]**2  # (state[0] - state_pos[0])**2
    # print(loss, "with x:")
    # loss += .1 * state[1]**2 + .1 * state[3]**2

    # add force loss
    # loss += .2 * (x_dot_orig + torch.sum(action, axis=1))**2

    if printout:
        # print(
        #     "x before",
        #     x_orig[0].item(),
        #     "x after",
        #     x[0].item(),  # "theta max possible", theta_max_possible[0].item()
        # )
        # print("losses:")
        print("theta", theta[0].item())
        print("position_loss", position_loss[0].item())
        print("vel_loss", vel_loss[0].item())
        # print("factor", factor[0].item())
        print("together", (factor * (position_loss + vel_loss) * .1)[0].item())
        print("angle loss", 13 * angle_loss[0].item())
        print()
    # print(fail)
    return torch.sum(loss)  # + angle_acc)
