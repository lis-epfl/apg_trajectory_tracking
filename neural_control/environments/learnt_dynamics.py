import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from neural_control.environments.flightmare_dynamics import (
    FlightmareDynamics
)


class LearntDynamics(nn.Module, FlightmareDynamics):

    def __init__(self, initial_params={}):
        FlightmareDynamics.__init__(self, modified_params=initial_params)
        super(LearntDynamics, self).__init__()

        # Action transformation parameters
        self.linear_at = nn.Parameter(
            torch.diag(torch.ones(4)), requires_grad=True
        )
        # non-linear transformation:
        # self.action_layer1 = nn.Parameter(
        #     torch.ones(4, 64), requires_grad=True
        # )
        # self.action_layer2 = nn.Parameter(
        #     torch.ones(64, 4), requires_grad=True
        # )

        # VARIABLES - dynamics parameters
        # self.torch_translational_drag = torch.Variable(torch.from_numpy(
        #     self.copter_params.translational_drag
        # ))
        # self.torch_rotational_drag = torch.Variable(torch.from_numpy(
        #     self.copter_params.rotational_drag
        # ))
        self.mass = nn.Parameter(
            torch.tensor([self.mass]),
            requires_grad=True  # , name="mass"
        )
        self.down_drag = nn.Parameter(
            torch.tensor([self.down_drag]).float(),
            requires_grad=True,
            # name="down_draf"
        )
        self.torch_inertia_vector = nn.Parameter(
            torch.from_numpy(self.inertia_vector).float(),
            requires_grad=True,
            # name="inertia"
        )
        self.torch_kinv_vector = nn.Parameter(
            torch.tensor(self.copter_params.kinv_ang_vel_tau).float(),
            requires_grad=True,
            # name="kinv"
        )

        # derivations from params
        self.torch_inertia_J = torch.diag(self.torch_inertia_vector)
        # self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector)
        self.torch_kinv_ang_vel_tau = torch.diag(self.torch_kinv_vector)

    def forward(self, state, action, dt):
        action_transformed = self.linear_at(action)
        # run through D1
        new_state = self.simulate_quadrotor(action_transformed, state, dt)
        # TODO: state transformation?
        return new_state
