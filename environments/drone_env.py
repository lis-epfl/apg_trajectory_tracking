"""
Adapted from https://github.com/ngc92/quadgym
"""
import math
import time
import pprint
from types import SimpleNamespace
import numpy as np
import torch

import gym
from gym import spaces
from gym.utils import seeding
from gym_quadrotor.dynamics import Euler
from gym_quadrotor.dynamics.coordinates import (
    angle_difference, angvel_to_euler
)

from gym_quadrotor.envs.rendering import Renderer, Ground, QuadCopter
try:
    from .copter import copter_params, DynamicsState
    from .drone_dynamics import simulate_quadrotor
except ImportError:
    from copter import copter_params, DynamicsState
    from drone_dynamics import simulate_quadrotor


class QuadRotorEnvBase(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    action_space = spaces.Box(0, 1, (4, ), dtype=np.float32)
    observation_space = spaces.Box(0, 1, (6, ), dtype=np.float32)

    def __init__(self, params: dict = None):

        # set up the renderer
        self.renderer = Renderer()
        self.renderer.add_object(Ground())
        self.renderer.add_object(QuadCopter(self))

        # set to supplied copter params, or to default value
        if params is None:
            params = copter_params
        self.setup = SimpleNamespace(**params)

        # just initialize the state to default, the rest will be done by reset
        self._state = DynamicsState()
        self.random_state = None
        self.seed()
        # added from motor one
        self._error_target = 1 * np.pi / 180
        self._in_target_reward = 0.5
        self._boundary_penalty = 1.0
        # self._attitude_reward = AttitudeReward(
        #     1.0, attitude_error_transform=np.sqrt
        # )

    @staticmethod
    def get_is_stable(np_state):
        attitude_condition = np.all(np.absolute(np_state[3:6] < .5))
        position_condition = 0 < np_state[2] < 5
        return attitude_condition and position_condition

    def step(self, action):
        action = np.clip(self._process_action(action), 0.0, 1.0)
        assert action.shape == (4, ), f"action not size 4 but {action.shape}"

        # set the blade speeds. as F ~ w², and we want F ~ action.
        torch_state = torch.from_numpy(np.array([self._state.as_np]))
        torch_action = torch.from_numpy(np.array([action]))
        new_state_arr = simulate_quadrotor(torch_action, torch_state)
        numpy_out_state = new_state_arr.numpy()[0]
        # update internal state
        self._state.from_np(numpy_out_state)
        is_stable = self.get_is_stable(numpy_out_state)
        # attitude = self._state.attitude

        # How this works: Check whether the angle is above the boundaries
        # (threshold), if yes, clip them (in the clip_attitude function) and
        # return whether it needed to be clipped, give penalty on reward if yes
        # if clip_attitude(self._state, np.pi / 4):
        #     reward -= self._boundary_penalty
        # resets the velocity after each step --> we don't want to do that
        # ensure_fixed_position(self._state, 1.0)

        return numpy_out_state, is_stable

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

            # update the renderer's center position
            self.renderer.set_center(self._state.position[0])

        return self.renderer.render(mode, close)

    def close(self):
        self.renderer.close()

    def reset(self, strength=.8):

        self._state = DynamicsState()
        # # possibility 1: reset to zero
        # zero_state = np.zeros(20)
        # zero_state[9:13] = 500
        # zero_state[2] = 1
        # self._state.from_np(zero_state)

        self.randomize_angle(5 * strength)
        self.randomize_angular_velocity(2.0)
        self._state.attitude.yaw = self.random_state.uniform(
            low=-0.3 * strength, high=0.3 * strength
        )

        self._state.position[2] = 1
        self.randomize_rotor_speeds(200, 500)
        # yaw control typically expects slower velocities
        self._state.angular_velocity[2] *= 0.5 * strength

        # self.renderer.set_center(None)

        return self._get_state()

    def get_copter_state(self):
        return self._state

    def _get_state(self):
        s = self._state
        rate = angvel_to_euler(s.attitude, s.angular_velocity)
        state = [
            s.attitude.roll, s.attitude.pitch,
            angle_difference(s.attitude.yaw, 0.0), rate[0], rate[1], rate[2]
        ]
        return np.array(state)

    def _process_action(self, action):
        return action

    # ------------ Functions to randomize the state -----------------------
    # env functions
    def seed(self, seed=None):
        self.random_state, seed = seeding.np_random(seed)
        return [seed]

    # utility functions
    def randomize_angle(self, max_pitch_roll: float):
        self._state._attitude = random_angle(self.random_state, max_pitch_roll)

    def randomize_velocity(self, max_speed: float):
        self._state.velocity[:] = self.random_state.uniform(
            low=-max_speed, high=max_speed, size=(3, )
        )

    def randomize_rotor_speeds(self, min_speed: float, max_speed: float):
        self._state.rotor_speeds[:] = self.random_state.uniform(
            low=min_speed, high=max_speed, size=(4, )
        )

    def randomize_angular_velocity(self, max_speed: float):
        self._state.angular_velocity[:] = self.random_state.uniform(
            low=-max_speed, high=max_speed, size=(3, )
        )

    def randomize_altitude(self, min_: float, max_: float):
        self._state.position[2] = self.random_state.uniform(
            low=min_, high=max_
        )


def random_angle(random_state: np.random.RandomState, max_pitch_roll: float):
    """
    Returns a random Euler angle where roll and pitch are limited to [-max_pitch_roll, max_pitch_roll].
    :param random_state: The random state used to generate the random numbers.
    :param max_pitch_roll: Maximum roll/pitch angle, in degrees.
    :return Euler: A new `Euler` object with randomized angles.
    """
    mpr = max_pitch_roll * math.pi / 180

    # small pitch, roll values, random yaw angle
    roll = random_state.uniform(low=-mpr, high=mpr)
    pitch = random_state.uniform(low=-mpr, high=mpr)
    yaw = random_state.uniform(low=-math.pi, high=math.pi)

    return Euler(roll, pitch, yaw)


# --------------------- Auxilary functions ----------------------


def clip_attitude(state: DynamicsState, max_angle: float):
    """
    Limits the roll and pitch angle to the given `max_angle`. 
    If roll or pitch exceed that angle,
    they are clipped and the angular velocity is set to 0.
    :param state: The quadcopter state to be modified.
    :param max_angle: Maximum allowed roll and pitch angle.
    :return: nothing.
    """
    attitude = state.attitude
    angular_velocity = state.angular_velocity
    clipped = False

    if attitude.roll > max_angle:
        attitude.roll = max_angle
        angular_velocity[:] = 0
        clipped = True
    if attitude.roll < -max_angle:
        attitude.roll = -max_angle
        angular_velocity[:] = 0
        clipped = True
    if attitude.pitch > max_angle:
        attitude.pitch = max_angle
        angular_velocity[:] = 0
        clipped = True
    if attitude.pitch < -max_angle:
        attitude.pitch = -max_angle
        angular_velocity[:] = 0
        clipped = True
    return clipped


def construct_states(num_data, episode_length=15):
    # data = np.load("data.npy")
    # assert not np.any(np.isnan(data))
    # return data
    env = QuadRotorEnvBase()
    data = []
    while len(data) < num_data:
        env.reset(strength=1)
        is_stable = True
        time_stable = 0
        while is_stable and time_stable < episode_length:
            # env.step(np.array([0, 0, 3, 0]))
            new_state, is_stable = env.step(np.random.rand(4))
            data.append(new_state)
            time_stable += 1
    data = np.array(data)
    return data


if __name__ == "__main__":
    env = QuadRotorEnvBase()
    # env = gym.make("QuadrotorStabilizeAttitude-MotorCommands-v0")

    for j in range(4):
        # print("reset: current state:")
        # pprint.pprint(env._state.formatted)
        # print()
        env.reset()
        # pprint.pprint(env._state.formatted)
        for i in range(20):
            # env.step(np.array([0, 0, 3, 0]))
            newstate = env.step(2 * np.random.rand(4))
            time.sleep(.2)
            # print(newstate)
            env.render()
        time.sleep(2)
