import numpy as np
from gym_quadrotor.dynamics import Euler

copter_params = {
    "thrust_factor": 5.723e-6,
    "drag_factor": 1.717e-7,
    "mass": 0.723,
    "rotational_drag": np.array([1, 1, 1]) * 1e-4,
    "translational_drag": np.array([1, 1, 1]) * 1e-4,
    "arm_length": 0.31,
    "rotor_inertia": 7.321e-5,
    # we assume a diagonal matrix
    "frame_inertia": np.array([8.678, 8.678, 32.1]) * 1e-3,
    "gravity": np.array([0.0, 0.0, -9.81]),
    "max_rotor_speed": 1000.0,
    "rotor_speed_half_time": 1.0 / 16,
}


class DynamicsState(object):

    def __init__(self):
        self._position = np.zeros(3)
        self._attitude = Euler(0.0, 0.0, 0.0)
        self._velocity = np.zeros(3)
        self._rotorspeeds = np.zeros(4)
        self._angular_velocity = np.zeros(3)

    def set_position(self, pos):
        self._position = pos

    @property
    def position(self):
        return self._position

    @property
    def attitude(self):
        return self._attitude

    @property
    def velocity(self):
        return self._velocity

    @property
    def rotor_speeds(self):
        return self._rotorspeeds

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @property
    def net_rotor_speed(self):
        return self._rotorspeeds[0] - self._rotorspeeds[1] + self._rotorspeeds[
            2] - self._rotorspeeds[3]

    @property
    def formatted(self):
        return {
            "position:": self._position,
            "attitude:": self._attitude,
            "velocity:": self._velocity,
            "rotorspeeds:": self._rotorspeeds,
            "angular_velocity:": self._angular_velocity
        }

    @property
    def as_np(self):
        """
        Convert state to np array
        """
        return np.array(
            (
                list(self._position) + list(self._attitude._euler) +
                list(self._velocity) + list(self._rotorspeeds) +
                list(self._angular_velocity)
            ),
            dtype=np.float32
        )

    def from_np(self, state_array):
        """
        Convert np array to dynamic state
        """
        self._position = state_array[:3]
        self._attitude = Euler(*tuple(state_array[3:6]))
        self._velocity = state_array[6:9]
        self._rotorspeeds = state_array[9:13]
        self._angular_velocity = state_array[13:16]
