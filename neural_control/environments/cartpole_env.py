"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import logging
import math
import numpy as np
import torch

import time
logger = logging.getLogger(__name__)
from neural_control.dynamics.cartpole_dynamics import (
    CartpoleDynamics, ImageCartpoleDynamics, SequenceCartpoleDynamics
)
try:
    import cv2
except ImportError:
    print("Warning: cv2 not installed, install for cartpole image experiments")
try:
    import neural_control.environments.cartpole_rendering as rendering
except:
    pass


class CartPoleEnv():
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dynamics, dt, thresh_div=.21):
        self.dynamics = dynamics
        self.is_img_dyn = isinstance(self.dynamics, ImageCartpoleDynamics)
        self.is_seq_dyn = isinstance(self.dynamics, SequenceCartpoleDynamics)

        self.dt = dt

        # Angle at which to fail the episode
        # self.theta_thresh = 12 * 2 * math.pi / 360
        self.thresh_div = thresh_div
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        self.state_limits = np.array([2.4, 7.5, np.pi, 7.5])

        self.viewer = None
        self.state = self._reset()

        self.steps_beyond_done = None

    def is_upright(self):
        theta = self.state[2]
        return theta > -self.thresh_div and theta < self.thresh_div

    def _step(
        self, action, image=None, state_action_buffer=None, is_torch=True
    ):
        torch_state = torch.tensor([list(self.state)])
        if not is_torch:
            action = torch.tensor([action])
        if self.is_img_dyn:
            next_torch_state = self.dynamics(
                torch_state, image, action, dt=self.dt
            )
        elif self.is_seq_dyn:
            next_torch_state = self.dynamics(
                torch_state, state_action_buffer, action, dt=self.dt
            )
        else:
            next_torch_state = self.dynamics(
                torch_state.float(), action, dt=self.dt
            )
        self.state = next_torch_state[0].numpy()
        # stay in bounds with theta
        theta = self.state[2]
        if theta > np.pi:
            self.state[2] = theta - 2 * np.pi
        if theta <= -np.pi:
            self.state[2] = 2 * np.pi + theta
        return self.state

    def _reset(self):
        # sample uniformly in the states
        # gauss = np.random.normal(0, 1, 4) / 2.5
        # gauss[0] = (np.random.rand(1) * 2 - 1) * self.x_threshold
        # gauss[2] = np.random.rand(1) * 2 - 1
        self.state = (np.random.rand(4) * 2 - 1) * self.state_limits
        # this achieves the same as below because then min is -0.05
        # self.np_random.uniform(low=-0.05, high=0.05, size=(4, ))
        self.steps_beyond_done = None
        return np.array(self.state)

    def _reset_upright(self):
        """
        reset state to a position of the pole close to the optimal upright pos
        """
        # randomize state between -0.25 and 0.25
        self.state = (np.random.rand(4) - .5) * .3
        # randomize theta between 0.15 max
        self.state[2] = (np.random.rand(1) - .5) * .1
        return self.state

    def _render(self, mode='human', close=False):
        """
        Drawing function to visualize pendulum - Use after each update!
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = (
                -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            )
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2, polewidth / 2, polelen - polewidth / 2,
                -polewidth / 2
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = (x[0] * scale + screen_width / 2.0) % 600  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def construct_states(
    num_data,
    dt,
    save_path="models/minimize_x/state_data.npy",
    thresh_div=.21,
    **kwargs
):
    # define parts of the dataset:
    randomized_runs = .8
    upper_balancing = 1
    one_direction = 1

    # Sample states
    dyn = CartpoleDynamics()
    env = CartPoleEnv(dyn, dt, thresh_div=thresh_div)
    data = []
    # randimized runs
    # while len(data) < num_data * randomized_runs:
    #     # run 100 steps then reset (randomized runs)
    #     for _ in range(10):
    #         action = np.random.rand() - 0.5
    #         state, _, _, _ = env._step(action)
    #         data.append(state)
    #     env._reset()

    # # after randomized runs: run balancing
    while len(data) < num_data:
        # only theta between -0.5 and 0.5
        env.state = (np.random.rand(4) - .5) * .1
        #  env.state[2] = (np.random.rand(1) - .5) * .2
        while env.is_upright():
            action = np.random.rand() - 0.5
            state = env._step(action, is_torch=False)
            data.append(state)
        env._reset()

    # # add one directional steps
    # while len(data) < num_data * one_direction:
    #     action = (-.5) * ((np.random.rand() > .5) * 2 - 1)
    #     for _ in range(30):
    #         state, _, fine, _ = env._step(action)
    #         data.append(state)
    #     env._reset()
    #
    data = np.array(data)

    # # sample only states, no sequences
    # state_limits = np.array([2.4, 5, np.pi, 5])
    # uniform_samples = np.random.rand(num_data, 4) * 2 - 1
    # data = uniform_samples * state_limits

    # print("generated random data:", data.shape)
    # eval_data = [data]  # augmentation: , data * (-1)
    # for name in os.listdir("data"):
    #     if name[0] != ".":
    #         eval_data.append(np.load(os.path.join("data", name)))
    # data = np.concatenate(eval_data, axis=0)
    # print("shape after adding evaluation data", data.shape)
    # save data optionally
    # if save_path is not None:
    #     np.save(save_path, data)
    return data[:num_data]


def preprocess_img(image, img_height, img_width):
    resized = cv2.resize(
        np.mean(image, axis=2),
        dsize=(img_height, img_width),
        interpolation=cv2.INTER_LINEAR
    )
    return ((255 - resized) > 0).astype(float)


def make_state_to_img_dataset(dataset_size=2000):
    dyn = CartpoleDynamics()
    env = CartPoleEnv(dyn, dt=0.1)

    min_theta, max_theta = (-0.3, 0.3)
    theta_range = max_theta - min_theta
    img_width_base = 300 // 2
    x_bound = 2.4
    img_des_half = 120 // 2
    min_x_diff, max_x_diff = (-1, 1)
    x_range = max_x_diff - min_x_diff

    inputs = np.zeros((dataset_size, 2))
    images = []
    for i in range(dataset_size):
        random_theta = np.random.rand() * theta_range + min_theta
        random_x = np.random.rand() * theta_range + min_theta
        inputs[i] = [random_x, random_theta]
        env.state = np.array([random_x, 0, random_theta, 0])
        img = preprocess_img(env._render(mode="rgb_array"), 300,
                             200)[75:175, img_width_base -
                                  img_des_half:img_width_base + img_des_half]

        images.append(img)
    np.savez("data/state_to_img.npz", inputs, np.array(images))


if __name__ == "__main__":
    make_state_to_img_dataset(2000)
