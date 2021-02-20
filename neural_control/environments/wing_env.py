import gym
import torch
import numpy as np

from neural_control.environments.wing_longitudinal_dynamics import (
    long_dynamics
)


class SimpleWingEnv(gym.Env):
    """
    Fixed wing drone environment
    """

    def __init__(self, dt):
        self.dt = dt
        self.reset()

    def zero_reset(self):
        self._state = np.array([0, 0, 10, 0, 0, 0])

    def reset(self):
        # no need to randomize because relative position used anyway
        x_pos = 0
        z_pos = 0
        # randomize around 10
        vel = np.random.rand(1) - .5 + 10
        vel_up = np.random.rand(1) - .5
        pitch_angle = np.deg2rad(np.random.rand(4) - 2)
        pitch_rate = np.random.rand(1) * 0.01 - 0.005

        self._state = np.array(
            [x_pos, z_pos, vel, vel_up, pitch_angle, pitch_rate]
        )

    def step(self, action):
        """
        action: tuple / list/np array of two values, between 0 and 1 (sigmoid)
        """
        thrust, del_e = action

        thrust_scaled = 1.3 + thrust * .4 - .2
        del_e_scaled = np.deg2rad(del_e * 10 - 5)

        action_torch = torch.tensor([[thrust_scaled, del_e_scaled]]).float()
        state_torch = torch.tensor([self._state.tolist()]).float()

        new_state = long_dynamics(state_torch, action_torch, self.dt)
        self._state = new_state[0].numpy()

        is_stable = True  # TODO
        return self._state, is_stable
