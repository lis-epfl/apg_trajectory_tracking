"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import logging
import math
import numpy as np
logger = logging.getLogger(__name__)
import time
from . import cartpole_rendering as rendering


class CartPoleEnv():
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 40.0  # prev 30 TODO
        self.tau = 0.02  # seconds between state updates
        self.muc = 0.0005
        self.mup = 0.000002

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        self.state_limits = np.array([2.4, 7.5, np.pi, 7.5])

        self.viewer = None
        self.state = self._reset()

        self.steps_beyond_done = None

    def _step(self, action):
        """
        Update state after action
        """
        # compute force
        force = self.force_mag * action
        # get state and compute next state
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        sig = self.muc * np.sign(x_dot)
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
        # TODO: swapped those! - is that okay?
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        if theta > np.pi:
            theta = theta - 2 * np.pi
        if theta <= -np.pi:
            theta = 2 * np.pi + theta
        assert np.abs(theta) <= np.pi, "theta greater pi"

        # change x such that it is not higher than 3
        x = ((x * 100 + 240) % 480 - 240) / 100
        assert np.abs(x) <= self.x_threshold
        self.state = (x, x_dot, theta, theta_dot)

        # Check whether still in feasible area etc
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            # if self.steps_beyond_done == 0:
            #     logger.warning(
            #         "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior."
            #     )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def _reset(self):
        # sample uniformly in the states
        # gauss = np.random.normal(0, 1, 4) / 2.5
        # gauss[0] = (np.random.rand(1) * 2 - 1) * self.x_threshold
        # gauss[2] = np.random.rand(1) * 2 - 1
        self.state = (np.random.rand(4) * 2 - 1) * self.state_limits
        # self.state = (np.random.rand(4) * 2 - 1) * self.state_limits
        # this achieves the same as below because then min is -0.05
        # self.np_random.uniform(low=-0.05, high=0.05, size=(4, ))
        self.steps_beyond_done = None
        return np.array(self.state)

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


def construct_states(num_data, save_path="models/minimize_x/state_data.npy"):
    # define parts of the dataset:
    randomized_runs = .8
    upper_balancing = 1
    one_direction = 1

    # Sample states
    env = CartPoleEnv()
    data = []
    # randimized runs
    while len(data) < num_data * randomized_runs:
        # run 100 steps then reset (randomized runs)
        for _ in range(10):
            action = np.random.rand() - 0.5
            state, _, _, _ = env._step(action)
            data.append(state)
        env._reset()

    # # after randomized runs: run balancing
    while len(data) < num_data:
        fine = False
        # only theta between -0.5 and 0.5
        # env.state = env.state * .5  # TODO
        env.state[2] = (np.random.rand(1) - .5) * .2
        while not fine:
            action = np.random.rand() - 0.5
            state, _, fine, _ = env._step(action)
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

    print("generated random data:", data.shape)
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


if __name__ == "__main__":
    data = np.load("../trained_models/minimize_x_best_model/state_data.npy")
    env = CartPoleEnv()
    pick_random_starts = np.random.permutation(len(data))[:100]
    for i in pick_random_starts:
        env.state = data[i]
        for j in range(10):
            env._step(np.random.rand(1) - .5)
            env._render()
            time.sleep(.1)
        time.sleep(1)
        # sign = np.sign(np.random.rand() - 0.5)
        # if i % 2 == 0:
        #     sign = -1
        # else:
        #     sign = 1

        # USUAL pipeline
        # action = 2 * (np.random.rand() - 0.5)
        # out = env._step(action)
        # print("action", action, "out:", out)
