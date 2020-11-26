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
max_force_mag = 10.0  # was 10 previously, but only action 1 and -1
# so now we want continous actions
tau = 0.02  # seconds between state updates
muc = 0.0005
mup = 0.000002
# Angle at which to fail the episode
theta_threshold_radians = 12 * 2 * torch.pi / 360
x_threshold = 2.4


def control_loss_function(action, state):

    # bring action into -1 1 range
    action = (torch.sigmoid(action) - .5) * 3

    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    # set the normalization
    # print(theta)
    # theta_prev = torch.maximum(theta, torch.zeros(theta.size()) + 0.001)
    theta_dot = state[:, 3]

    force = max_force_mag * action
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    sig = muc * torch.sign(x_dot)
    temp = force + polemass_length * theta_dot * theta_dot * sintheta
    thetaacc = (
        gravity * sintheta - (costheta * (temp - sig)) -
        (mup * theta_dot / polemass_length)
    ) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))
    # not required to compute next position of x
    # xacc = (
    #     temp - (polemass_length * thetaacc * costheta) - sig
    # ) / total_mass
    # x_dot = x_dot + tau * xacc
    # x = x + tau * x_dot
    # TODO: swapped these two
    theta_dot = theta_dot + tau * thetaacc
    theta = theta + tau * theta_dot

    loss = torch.sum((theta - target_state)**2)
    # Variable(torch.sum((theta - target_state)**2), requires_grad=True)
    return loss
