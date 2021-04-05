import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from neural_control.environments.flightmare_dynamics import (
    FlightmareDynamics
)


class LearntDynamics(nn.Module, FlightmareDynamics):

    def __init__(self, initial_params={}):
        FlightmareDynamics.__init__(self, **initial_params)
        super(LearntDynamics, self).__init__()

        # Action transformation parameters
        self.linear_at = nn.Parameter(
            torch.diag(torch.ones(4)), requires_grad=True
        )
        self.linear_state_1 = nn.Linear(16, 64)
        torch.nn.init.constant_(self.linear_state_1.weight, 0)
        torch.nn.init.constant_(self.linear_state_1.bias, 0)

        self.linear_state_2 = nn.Linear(64, 12)
        torch.nn.init.constant_(self.linear_state_2.weight, 0)
        torch.nn.init.constant_(self.linear_state_2.bias, 0)

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
            torch.tensor(self.kinv_ang_vel_tau).float(),
            requires_grad=True,
            # name="kinv"
        )

        # derivations from params
        self.torch_inertia_J = torch.diag(self.torch_inertia_vector)
        # self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector)
        self.torch_kinv_ang_vel_tau = torch.diag(self.torch_kinv_vector)

    def state_transformer(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        layer_1 = torch.relu(self.linear_state_1(state_action))
        new_state = self.linear_state_2(layer_1)
        # TODO: activation function?
        return new_state

    def forward(self, action, state, dt):
        action_transformed = torch.matmul(
            self.linear_at, torch.unsqueeze(action, 2)
        )[:, :, 0]
        # run through D1
        new_state = self.simulate_quadrotor(action_transformed, state, dt)
        # run through T
        added_new_state = self.state_transformer(state, action_transformed)
        return new_state + added_new_state
