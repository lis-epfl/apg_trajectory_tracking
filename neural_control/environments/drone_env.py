"""
Adapted from https://github.com/ngc92/quadgym
"""
import math
import time
from types import SimpleNamespace
import numpy as np
import torch

import gym
from gym import spaces
from gym.utils import seeding

from neural_control.utils.straight import (
    straight_training_sample, get_reference
)
from neural_control.environments.rendering import (
    Renderer, Ground, QuadCopter
)
from neural_control.environments.copter import (
    copter_params, DynamicsState, Euler
)
from neural_control.environments.drone_dynamics import (
    simulate_quadrotor, linear_dynamics
)
from neural_control.utils.generate_trajectory import generate_trajectory

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QuadRotorEnvBase(gym.Env):
    """
    Simple simulation environment for a drone
    Drone parameters are defined in file copter.py (copter_params dict)
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    action_space = spaces.Box(0, 1, (4, ), dtype=np.float32)
    observation_space = spaces.Box(0, 1, (6, ), dtype=np.float32)

    def __init__(self, dt):

        # set up the renderer
        self.renderer = Renderer()
        self.renderer.add_object(Ground())
        self.renderer.add_object(QuadCopter(self))

        # set to supplied copter params, or to default value
        self.setup = SimpleNamespace(**copter_params)

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
        self.dt = dt

    @staticmethod
    def get_is_stable(np_state, thresh=.4):
        """
        Return for a given state whether the drone is stable or failure
        Returns bool --> if true then still stable
        """
        # only roll and pitch are constrained
        attitude_condition = np.all(np.absolute(np_state[3:5]) < thresh)
        return attitude_condition

    def get_acceleration(self):
        """
        Compute acceleration from current state (pos, vel and att)
        """
        acc = (self._state.velocity - self._state._last_velocity) / self.dt
        return acc

    def step(self, action, thresh=.4):
        """
        Apply action to the current drone state
        Returns:
            New state as np array
            bool indicating whether drone failed
        """
        action = np.clip(self._process_action(action), 0.0, 1.0)
        assert action.shape == (4, ), f"action not size 4 but {action.shape}"

        # set the blade speeds. as F ~ w², and we want F ~ action.
        torch_state = torch.from_numpy(np.array([self._state.as_np])
                                       ).to(device)
        torch_action = torch.from_numpy(np.array([action])).float().to(device)
        new_state_arr = simulate_quadrotor(
            torch_action, torch_state, dt=self.dt
        )
        numpy_out_state = new_state_arr.cpu().numpy()[0]
        # update internal state
        self._state.from_np(numpy_out_state)
        # attitude = self._state.attitude

        # How this works: Check whether the angle is above the boundaries
        # (threshold), if yes, clip them (in the clip_attitude function) and
        # return whether it needed to be clipped, give penalty on reward if yes
        # if clip_attitude(self._state, np.pi / 4):
        #     reward -= self._boundary_penalty
        # resets the velocity after each step --> we don't want to do that
        # ensure_fixed_position(self._state, 1.0)

        return numpy_out_state, self.get_is_stable(
            numpy_out_state, thresh=thresh
        )

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

            # update the renderer's center position
            self.renderer.set_center(0)  # self._state.position[0])

        return self.renderer.render(mode, close)

    def close(self):
        self.renderer.close()

    def zero_reset(self, position_x=0, position_y=0, position_z=2):
        """
        Reset to easiest state: zero velocities and attitude and given position
        Arguments:
            Position (in three floats)
        """
        self.reset()
        state_arr = self._state.as_np
        state_arr[:3] = [position_x, position_y, position_z]
        # set attitude and angular vel to zero
        state_arr[3:] = 0
        self._state._velocity = np.zeros(3)
        self._state.from_np(state_arr)
        return self._state.as_np

    def render_reset(self, strength=.8):
        """
        Reset to a random state, but require z position to be at least 1
        """
        self.reset(strength=strength)
        self._state.position[2] += 2

    def reset(self, strength=.8):
        """
        Reset drone to a random state
        """
        self._state = DynamicsState()
        # # possibility 1: reset to zero
        #

        self.randomize_angle(3 * strength)
        self.randomize_angular_velocity(2.0 * strength)
        self._state.attitude.yaw = self.random_state.uniform(
            low=-1.5, high=1.5
        )
        self._state.position[:3] = np.random.rand(3) * 2 - 1
        # self.randomize_rotor_speeds(200, 500)
        # yaw control typically expects slower velocities
        self._state.angular_velocity[2] *= 0.5  # * strength
        self.randomize_velocity(3)

        # self.renderer.set_center(None)

        return self._state

    def get_copter_state(self):
        return self._state

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
        self._state._last_velocity = self._state.velocity.copy()

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


def random_angle(random_state, max_pitch_roll):
    """
    Returns a random Euler angle
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

# data = np.load("training_data.npy")
# rand_inds = np.random.permutation(len(data))
# # remove problematic ones
# rand_inds = (rand_inds[np.logical_and(rand_inds % 501 < 490,
#             rand_inds % 501 > 30)])


def load_training_data(len_data, ref_length=5, reset_strength=0, **kwargs):
    np.set_printoptions(precision=2, suppress=True)
    # print((np.random.rand(3) - .5) * reset_strength)
    some_point = np.random.randint(0, len(rand_inds) - len_data, 1)[0]
    use_start_inds = rand_inds[some_point:some_point + len_data]
    drone_states, ref_states = [], []
    for start in use_start_inds:
        next_ref = np.array(data[start + 1:start + 1 + ref_length])
        ref_states.append(next_ref)
        # drone_states.append(data[start])
        # noise_applied = (
        #     np.ones(12) + reset_strength * (np.random.rand(12) - .5)
        # )
        # print("not noisy drone state", data[start])
        random_direction = np.random.rand(12) - .5
        div_vector = (
            next_ref[1] - next_ref[0]
        ) * random_direction * reset_strength
        # print("div_vector", div_vector)
        noisy_drone_state = data[start] + div_vector
        # noisy_drone_state[:3] += (np.random.rand(3) - .5) * reset_strength

        drone_states.append(noisy_drone_state)

        # print(noisy_drone_state)
        # print("ref_states")
        # print(ref_states)
        # print()

    drone_states = np.array(drone_states)
    ref_states = np.array(ref_states)
    return drone_states[:len_data], ref_states[:len_data]


def full_state_training_data(
    len_data, ref_length=5, reset_strength=0, dt=0.02, **kwargs
):
    """
    Use trajectory generation of Elia to generate random trajectories and then
    position the drone randomly around the start
    Arguments:
        reset_strength: how much the drone diverges from its desired state
    """
    sample_freq = ref_length  # * some number
    # TODO: might want to sample less frequently
    drone_states = np.zeros((len_data + 200, 12))
    ref_states = np.zeros((len_data + 200, ref_length, 12))

    counter = 0
    while counter < len_data:
        traj = generate_trajectory(10, dt)  # TODO: freq of trajectory?
        traj_cut = traj[:-ref_length]
        # select every xth sample as the current drone state
        selected_starts = traj_cut[::sample_freq, :]
        nr_states_added = len(selected_starts)
        # add drone states
        drone_states[counter:counter + nr_states_added, :] = selected_starts
        # add ref states
        for i in range(ref_length):
            ref_states[counter:counter + nr_states_added + 1,
                       i] = traj[i::sample_freq]
        counter += nr_states_added

    return drone_states[:len_data], ref_states[:len_data]


def trajectory_training_data(
    len_data,
    max_drone_dist=0.1,
    ref_length=5,
    reset_strength=1,
    load_selfplay=None,  # "data/jan_2021.npy",
    dt=0.02,
    **kwargs
):
    """
    Generate training dataset for trajectories as input
    Arguments:
        len_data: int, how much data to generate
        max_drone_dist: Maximum distance of the drone from the first state
        ref_length: Number of states sampled from reference traj
        reset_strength: How much to reset the model
        load_selfplay: file path where data from self play is located
    Returns:
        Array of size (len_data, reference_shape) with the training data
    """
    if load_selfplay is not None:
        # load training data from np array
        data = np.load(load_selfplay)
        rand_inds = np.random.permutation(len(data))
        ind_counter = 0
    env = QuadRotorEnvBase(dt)
    drone_states, ref_states = [], []
    for _ in range(len_data):
        if load_selfplay is not None and np.random.rand() < .5:
            drone_state = data[rand_inds[ind_counter]]
            ind_counter += 1
        else:
            env.reset(strength=reset_strength)
            # sample a drone state
            drone_state = env._state.as_np
        pos0, vel0 = (drone_state[:3], drone_state[6:9])
        acc0 = np.random.rand(3) * 2 - 1

        # sample a direction where the next position is located
        norm_vel = np.linalg.norm(vel0)
        # should not diverge too much from velocity direction of drone
        sampled_pos_dir = (
            vel0 + (np.random.rand(3) - .5) * reset_strength * norm_vel
        )
        sampled_pos_dir = sampled_pos_dir / np.linalg.norm(sampled_pos_dir)
        # sample direction where the velocity should show in the end
        sampled_vel_dir = (
            sampled_pos_dir + (np.random.rand(3) - .5) * reset_strength
        )

        # final goal state:
        posf = pos0 + sampled_pos_dir * max_drone_dist
        # multiply by norm to get a velocity of similar strength
        velf = sampled_vel_dir * (norm_vel * (1 + np.random.rand(1) * .2 - .1))

        reference_states = get_reference(
            pos0,
            vel0,
            acc0,
            posf,
            velf,
            ref_length=ref_length,
            delta_t=env.dt
        )

        drone_states.append(drone_state)
        ref_states.append(reference_states)
    drone_states = np.array(drone_states)
    ref_states = np.array(ref_states)
    # np.save("ref_states.npy", ref_states)
    # exit()
    return drone_states, ref_states


if __name__ == "__main__":
    env = QuadRotorEnvBase(0.02)
    env.reset()
    states, ref = full_state_training_data(1000)
    # np.save("drone_states.npy", a1)
    # np.save("ref_states.npy", a2)
    # env = gym.make("QuadrotorStabilizeAttitude-MotorCommands-v0")
    # states = np.load("check_added_data.npy")
    print(np.mean(states[:, :6], axis=0))
    print(np.std(states[:, :6], axis=0))
    #  np.load("data_backup/collected_data.npy")
    for j in range(100):
        print([round(s, 2) for s in states[j, :6]])
        env._state.from_np(states[j])
        env.render()
        time.sleep(.1)
