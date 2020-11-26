import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# target state means that theta is zero --> only third position matters
# TODO: can check at a later stage whether it makes sense to also have theta
# dot equal to zero (rather no velocity)
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))


class ControlLoss(nn.Module):

    def __init__(self):
        super(ControlLoss, self).__init__()
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.max_force_mag = 10.0  # was 10 previously, but only action 1 and -1
        # so now we want continous actions
        self.tau = 0.02  # seconds between state updates
        self.muc = 0.0005
        self.mup = 0.000002
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * torch.pi / 360
        self.x_threshold = 2.4

    def forward(self, action, state):
        """
        state: tensor size [?, 4] with x, x_dot, theta, theta_dot
        action: size [?] float between -1 and 1
        """
        # x, x_dot, theta, theta_dot = state
        x = state[:, 0]
        x_dot = state[:, 1]
        theta = state[:, 2]

        # set the normalization
        theta_prev = torch.maximum(theta, torch.zeros(theta.size()) + 0.001)

        theta_dot = state[:, 3]
        force = self.max_force_mag * action
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        sig = self.muc * torch.sign(x_dot)
        temp = force + self.polemass_length * theta_dot * theta_dot * sintheta
        thetaacc = (
            self.gravity * sintheta - (costheta * (temp - sig)) -
            (self.mup * theta_dot / self.polemass_length)
        ) / (
            self.length * (
                4.0 / 3.0 -
                self.masspole * costheta * costheta / self.total_mass
            )
        )
        xacc = (
            temp - (self.polemass_length * thetaacc * costheta) - sig
        ) / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)

        loss = Variable(
            torch.sum((theta / theta_prev - target_state)**2),
            requires_grad=True
        )
        return loss
