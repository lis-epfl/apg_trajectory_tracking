import gym
import torch
import numpy as np

from neural_control.environments.wing_longitudinal_dynamics import (
    long_dynamics
)
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
        self.renderer = Renderer()
        self.renderer.add_object(Ground())
        self.drone_render_object = FixedWingDrone(self)
        self.renderer.add_object(self.drone_render_object)

    def zero_reset(self):
        self._state = np.array([0, 0, 10, 0, 0, 0])

    def reset(self):
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
        thrust, del_e = action

        action_torch = torch.tensor([[thrust, del_e]]).float()
        state_torch = torch.tensor([self._state.tolist()]).float()

        new_state = long_dynamics(state_torch, action_torch, self.dt)
        self._state = new_state[0].numpy()

        is_stable = True  # TODO
        return self._state, is_stable

    def render(self, mode='human', close=False):
        if not close:
            self.renderer.setup()

            # update the renderer's center position
            self.renderer.set_center(0)

        return self.renderer.render(mode, close)

    def close(self):
        self.renderer.close()


def run_wing_flight(num_traj=100, traj_len=1000, dt=0.01, **kwargs):
    sampled_trajectories = []
    for i in range(num_traj):
        env = SimpleWingEnv(dt)
        env.zero_reset()
        sampled_states = []
        for j in range(traj_len):
            if j % 100 == 0:
                # always keep same action for 10 steps
                T = np.random.rand(1)[0]
                del_e = np.random.rand(1)[0]
            # 10 + np.random.rand(1) * 4 - 1
            # del_e = np.random.rand(1) * 10 - 5
            new_state, _ = env.step((T, del_e))
            sampled_states.append(new_state)
        sampled_trajectories.append(np.array(sampled_states))
    return np.array(sampled_trajectories)


def sample_training_data(
    num_samples, num_points_per_traj=20, len_per_trajectory=350, **kwargs
):
    # training data: only a state and a position --> one that is reachable
    start_way, end_way = (200, 350)
    start_state, end_state = (0, 150)

    # compute number of trajectories required given the above
    num_flights = int(num_samples / num_points_per_traj)

    sampled_trajectories = run_wing_flight(
        num_traj=num_flights, traj_len=len_per_trajectory, **kwargs
    )

    training_states = []
    training_refs = []
    for i in range(len(sampled_trajectories)):
        current_flight = sampled_trajectories[i]
        # for each trajectory, sample x states
        rand_perm = np.random.permutation(np.arange(start_state, end_state, 1))
        take_states = rand_perm[:num_points_per_traj]
        # for each trajectory, sample x reference points
        rand_perm = np.random.permutation(np.arange(start_way, end_way, 1))
        take_way = rand_perm[:num_points_per_traj]
        # print(take_states, take_way)
        # print(current_flight[take_states].shape)
        training_states.extend(current_flight[take_states].tolist())
        training_refs.extend(current_flight[take_way, :2].tolist())
    training_states = np.array(training_states)
    training_refs = np.array(training_refs)
    return training_states, training_refs


if __name__ == "__main__":
    states, refs = sample_training_data(100)
    print(states.shape, refs.shape)
