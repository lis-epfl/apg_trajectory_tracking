import gym
import torch
import numpy as np

from neural_control.environments.wing_3D_dynamics import (long_dynamics)
from neural_control.environments.rendering import (
    Renderer, Ground, RenderedObject, FixedWingDrone
)


class SimpleWingEnv(gym.Env):
    """
    Fixed wing drone environment
    """

    def __init__(self, dt):
        self.dt = dt
        self.reset()
        self.renderer = Renderer(viewer_shape=(1000, 500), y_axis=7)
        self.renderer.add_object(Ground())
        self.drone_render_object = FixedWingDrone(self)
        self.renderer.add_object(self.drone_render_object)

    def zero_reset(self):
        self._state = np.zeros(12)
        self._state[3] = 11.5

    def reset(self):
        # TODO: wrong at the moment
        # no need to randomize because relative position used anyway
        x_pos = 0
        z_pos = 0
        # randomize around 10
        vel = np.random.rand(1) - .5 + 10
        vel_up = np.random.rand(1) - .5
        pitch_angle = np.deg2rad(np.random.rand(1) * 4 - 2)
        pitch_rate = np.random.rand(1) * 0.01 - 0.005

        self._state = np.array(
            [x_pos, z_pos, vel[0], vel_up[0], pitch_angle[0], pitch_rate[0]]
        )

    def step(self, action):
        """
        action: tuple / list/np array of two values, between 0 and 1 (sigmoid)
        """

        action_torch = torch.tensor([action.tolist()]).float()
        state_torch = torch.tensor([self._state.tolist()]).float()

        new_state = long_dynamics(state_torch, action_torch, self.dt)
        self._state = new_state[0].numpy()

        is_stable = np.all(np.absolute(self._state[6:8]) < .7)
        # if not is_stable:
        #     print("unstable!", self._state[6:9])
        return self._state, is_stable

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

            # update the renderer's center position
            self.renderer.set_center(0)

        return self.renderer.render(mode, close)

    def close(self):
        self.renderer.close()


def run_wing_flight(env, traj_len=1000, render=0, **kwargs):

    # define action prior here
    action_prior = np.array([.25, .5, .5, .5])
    env.zero_reset()
    sampled_states = []
    for j in range(traj_len):
        if j % 100 == 0:
            # always keep same action for 10 steps
            sampled_action = np.random.normal(scale=.02, size=4)
            scaled_action = np.clip(sampled_action + action_prior, 0, 1)
            # print("ACTION")
            # print(scaled_action)
            # scaled_action = np.array([1.9 / 7, 0.5, 0.5, 0.5])
        new_state, stable = env.step(scaled_action)
        # np.set_printoptions(suppress=1, precision=3)
        # print(new_state[:3])
        if not stable:
            break
        if render:
            env.render()
        sampled_states.append(new_state)
    return np.array(sampled_states)


def generate_unit_vecs(num_vecs, mean_vec=[1, 0, 0], std=.15):
    """
    Generate unit vectors that are normal distributed around mean_vec
    """
    gauss_vecs = np.random.multivariate_normal(
        mean_vec, [[std, 0, 0], [0, std, 0], [0, 0, std]], size=num_vecs
    )
    gauss_vecs[gauss_vecs[:, 0] < 0.01, 0] = 1
    # gauss_vecs = np.array(
    #     [vec / np.linalg.norm(vec) for vec in gauss_vecs if vec[0] > 0]
    # )
    return gauss_vecs


def sample_training_data(
    num_samples, dt=0.01, take_every=10, traj_len=1000, vec_std=.15, **kwargs
):
    """
    Fly some trajectories in order to sample drone states
    Then add random unit vectors in all directions
    """
    # sample random direction vectors
    gauss_vecs = generate_unit_vecs(num_samples, std=vec_std)
    env = SimpleWingEnv(dt)

    # combine states and gauss_vecs
    training_states = []
    training_refs = []
    counter = 0
    leftover = np.inf
    while leftover > 0:
        # sample trajectory
        traj = run_wing_flight(env, traj_len=200, **kwargs)
        # sample states from trajectory
        nr_samples = min([len(traj) // take_every, leftover])
        for i in range(nr_samples):
            # don't start at zero each time
            curr_ind = int(i * take_every + np.random.rand() * 5)
            drone_state = traj[curr_ind]

            # sample gauss vec
            drone_ref = drone_state[:3] + gauss_vecs[counter]
            # print(gauss_vecs[counter] / np.linalg.norm(gauss_vecs[counter]))
            training_states.append(drone_state)
            training_refs.append(drone_ref)
            counter += 1
        leftover = num_samples - len(training_refs)
    # make arrays
    training_states = np.array(training_states)
    training_refs = np.array(training_refs)
    return training_states, training_refs


if __name__ == "__main__":
    # states, refs = sample_training_data(1000)
    # print(states.shape, refs.shape)
    # np.save("states.npy", states)
    # np.save("ref.npy", refs)
    env = SimpleWingEnv(0.05)
    run_wing_flight(env, traj_len=1000, dt=0.01, render=1)
