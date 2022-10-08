import gym
import math
import numpy as np
import torch
import time
from gym import spaces
import cv2
from gym.utils import seeding

from neural_control.environments.cartpole_env import CartPoleEnv
from neural_control.environments.wing_env import SimpleWingEnv
from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.trajectory.q_funcs import project_to_line
from neural_control.dataset import WingDataset, QuadDataset
from neural_control.trajectory.generate_trajectory import (
    load_prepare_trajectory
)
from neural_control.trajectory.random_traj import PolyObject
metadata = {'render.modes': ['human']}

buffer_len = 3
img_width, img_height = (200, 300)
crop_width = 60
center_at_x = True


class CartPoleEnvRL(gym.Env, CartPoleEnv):

    def __init__(self, dynamics, dt=0.05, **kwargs):
        CartPoleEnv.__init__(self, dynamics, dt=dt)

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # CHANGE HERE WHETHER image or not
        self.image_obs = True

        high = np.array(
            [
                self.x_threshold * 2, 20, self.theta_threshold_radians * 2, 20,
                1
            ] * buffer_len
        )
        self.thresh_div = 0.21
        self.action_space = spaces.Box(low=-1, high=1, shape=(1, ))

        if self.image_obs:
            high = np.ones((buffer_len, 100, 120))
            self.observation_space = spaces.Box(np.zeros(high.shape), high)
        else:
            self.obs_dim = len(high)
            self.observation_space = spaces.Box(-high, high)
        self.init_buffers()

    def init_buffers(self):
        self.state_buffer = np.zeros((buffer_len, 4))
        self.action_buffer = np.zeros((buffer_len, 1))
        self.image_buffer = np.zeros((buffer_len, 100, 120))

    def set_state(self, state):
        self.state = state
        self._state = state

    def _convert_image_buffer(self, state, crop_width=crop_width):
        # image and corresponding state --> normalize x pos in image buffer!
        img_width_half = self.image_buffer.shape[2] // 2
        if center_at_x:
            x_pos = state[0] / self.x_threshold
            x_img = int(img_width_half + x_pos * img_width_half)
            use_img_buffer = np.roll(
                self.image_buffer.copy(), img_width_half - x_img, axis=2
            )
            return use_img_buffer[:, 75:175, img_width_half -
                                  crop_width:img_width_half + crop_width]
        else:
            x_img = img_width_half
            return self.image_buffer[:, 75:175,
                                     x_img - crop_width:x_img + crop_width]

    def get_reward(self):
        survive_reward = 0.1
        angle_reward = .5 * (1.5 - abs(self.state[2]))
        vel_reward = .4 * (1.5 - abs(self.state[1]))
        return survive_reward + angle_reward + vel_reward

    def get_history_obs(self):
        state_action_history = np.concatenate(
            (self.state_buffer, self.action_buffer), axis=1
        )
        obs = np.reshape(state_action_history, (self.obs_dim))
        return obs

    def get_img_obs(self):
        new_img = self._render(mode="rgb_array")
        self.image_buffer = np.roll(self.image_buffer, 1, axis=0)
        self.image_buffer[0] = self._preprocess_img(new_img)
        return self._convert_image_buffer(self.state)

    def step(self, action):
        super()._step(action, is_torch=False)
        # print(self.state)
        done = not self.is_upright() or self.step_ind > 250

        # this reward is positive if theta is smaller 0.1 and else negative
        if not done:
            # training to stay stable with low velocity
            reward = 1.0 - abs(self.state[1])
        else:
            reward = 0.0

        info = {}
        self.step_ind += 1

        # update state buffer with new state
        self.state_buffer = np.roll(self.state_buffer, 1, axis=0)
        self.state_buffer[0] = self.state.copy()
        self.action_buffer = np.roll(self.action_buffer, 1, axis=0)
        self.action_buffer[0] = action.copy()
        if self.image_obs:
            self.obs = self.get_img_obs()
        else:
            self.obs = self.get_history_obs()

        return self.obs, reward, done, info

    def _preprocess_img(self, image):
        resized = cv2.resize(
            np.mean(image, axis=2),
            dsize=(img_height, img_width),
            interpolation=cv2.INTER_LINEAR
        )
        return ((255 - resized) > 0).astype(float)

    def reset(self):
        super()._reset_upright()
        for i in range(buffer_len):
            self.state_buffer[i] = self.state
        self.action_buffer = np.zeros(self.action_buffer.shape)
        self.step_ind = 0

        if self.image_obs:
            start_img = self._preprocess_img(self._render(mode="rgb_array"))
            self.image_buffer = np.array(
                [start_img for _ in range(buffer_len)]
            )
            return self.get_img_obs()
        else:
            return self.get_history_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        self._render(mode=mode)

    def close(self):
        if self.viewer:
            self.viewer.close()


class QuadEnvRL(QuadRotorEnvBase, gym.Env):

    def __init__(self, dynamics, dt, speed_factor=.2, nr_actions=10, **kwargs):
        self.dt = dt
        self.speed_factor = speed_factor
        self.nr_actions = nr_actions

        QuadRotorEnvBase.__init__(self, dynamics, dt)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4, ))

        # state and reference
        self.state_inp_dim = 15
        self.obs_dim = self.state_inp_dim + nr_actions * 9
        high = np.array([10 for _ in range(self.obs_dim)])
        self.observation_space = spaces.Box(
            -high, high, shape=(self.obs_dim, )
        )

        self.thresh_stable = 1.5
        self.thresh_div = .3

        kwargs["dt"] = dt
        kwargs['speed_factor'] = speed_factor
        kwargs["self_play"] = 0
        self.dataset = QuadDataset(1, **kwargs)

    def prepare_obs(self):
        obs_state, _, obs_ref, _ = self.dataset.prepare_data(
            self.state.copy(),
            self.current_ref[self.current_ind + 1:self.current_ind +
                             self.nr_actions + 1].copy()
        )
        return obs_state, obs_ref

    def state_to_obs(self):
        # get from dataset
        obs_state, obs_ref = self.prepare_obs()
        # flatten obs ref
        obs_ref = obs_ref.reshape((-1, self.obs_dim - self.state_inp_dim))
        # concatenate relative position and observation
        obs = torch.cat((obs_ref, obs_state), dim=1)[0].numpy()
        return obs

    def reset(self, test=0):
        # load random trajectory from train
        self.current_ref = load_prepare_trajectory(
            "data/traj_data_1", self.dt, self.speed_factor, test=test
        )
        self.renderer.add_object(PolyObject(self.current_ref))

        self.state = np.zeros(12)
        self.state[:3] = self.current_ref[0, :3]
        self._state.from_np(self.state)

        self.current_ind = 0
        self.obs = self.state_to_obs()
        return self.obs

    def get_divergence(self):
        return np.linalg.norm(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )

    def get_reward_mpc(self, action):
        """
        MPC type cost function turned into reward
        """
        pos_factor = 10
        u_thrust_factor = 5
        u_rates_factor = 0.1
        av_factor = 0.1
        vel_factor = 1

        pos_div = np.linalg.norm(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )
        pos_rew = self.thresh_div - pos_div
        vel_div = np.linalg.norm(
            self.current_ref[self.current_ind, 6:9] - self.state[6:9]
        )
        vel_rew = self.thresh_div - vel_div  # How high is velocity diff?
        u_rew = .5 - np.absolute(.5 - action)
        # have to use abs because otherwise not comparable to thresh div
        av_rew = np.sum(self.thresh_stable - (np.absolute(self.state[9:12])))

        # print()
        reward = .1 * (
            pos_factor * pos_rew + vel_factor * vel_rew + av_factor * av_rew +
            u_rates_factor * np.sum(u_rew[1:]) + u_thrust_factor * u_rew[0]
        )

        return reward

    def get_reward_mario(self, action):
        """
        ori_coeff: -0.01        # reward coefficient for orientation
        ang_vel_coeff: 0   # reward coefficient for angular velocity
        # epsilon coefficient
        pos_epsilon: 2        # reward epsilon for position 
        ori_epsilon: 0.2        # reward epsilon for orientation
        lin_vel_epsilon: 2   # reward epsilon for linear velocity
        ang_vel_epsilon: 0.2   # reward epsilon for angular velocity
        """
        pos_coeff = -0.02
        pos_epsilon = 2
        lin_vel_coeff = -0.002
        lin_vel_epsilon = 2
        survive_reward = 0.1  #  0.001 mario
        act_coeff = -0.001
        ori_coeff = -0.01  # ori_coeff: -0.01
        omega_coefficient = -0.001
        omega_epsilon = 2
        ori_epsilon = .2

        # position
        pos_loss = np.sum(
            self.current_ref[self.current_ind, :3] - self.state[:3]
        )**2
        pos_reward = pos_coeff * (pos_loss - pos_epsilon)
        # orientation:
        ori_loss = np.sum(
            self.current_ref[self.current_ind, 3:6] - self.state[3:6]
        )**2
        ori_reward = ori_coeff * (ori_loss - ori_epsilon)
        # velocity
        vel_loss = np.sum(
            self.current_ref[self.current_ind, 6:9] - self.state[6:9]
        )**2
        vel_reward = lin_vel_coeff * (vel_loss - lin_vel_epsilon)
        # body rates
        # omega_loss = np.sum(
        #     self.current_ref[self.current_ind, 9:] - self.state[9:]
        # )**2
        # omega_reward = omega_coefficient * (omega_loss - omega_epsilon)
        # action
        act_reward = act_coeff * np.sum((.5 - action)**2)

        # print(
        #     "pos", pos_reward, "vel", vel_reward, "survive", survive_reward,
        #     "act", act_reward, "ori", ori_reward, "omega", omega_reward
        # )
        return (
            pos_reward + vel_reward + survive_reward + act_reward +
            ori_reward  # + omega_reward
        )

    def set_state(self, state):
        self.state = state
        self._state.from_np(state)

    def step(self, action):
        # rescale action
        action = (action + 1) / 2
        self.state, is_stable = QuadRotorEnvBase.step(
            self, action, thresh=self.thresh_stable
        )
        self.obs = self.state_to_obs()
        self.current_ind += 1

        pos_div = self.get_divergence()

        done = (
            (not is_stable) or pos_div > self.thresh_div
            or self.current_ind > len(self.current_ref) - self.nr_actions - 2
        )

        reward = 0
        if not done:
            # reward = self.thresh_div - pos_div
            # reward = self.get_reward(action)
            reward = self.get_reward_mario(action)
        info = {}
        # print()
        # np.set_printoptions(precision=3, suppress=1)
        # # print(self.current_ref[:3, :3])
        # print(
        #     self.current_ind, self.state[:3],
        #     self.current_ref[self.current_ind, :3]
        # )
        # print(self.state)
        # print(self.obs.shape)
        # print(div, reward)
        return self.obs, reward, done, info

    def render(self, mode="human"):
        self._state.position[2] += 1
        QuadRotorEnvBase.render(self, mode=mode)
        self._state.position[2] -= 1


class WingEnvRL(gym.Env, SimpleWingEnv):

    def __init__(self, dynamics, dt, **kwargs):
        SimpleWingEnv.__init__(self, dynamics, dt)
        self.action_space = spaces.Box(low=0, high=1, shape=(4, ))

        obs_dim = 12
        # high = np.array([20 for k in range(obs_dim)])
        high = np.array([20, 20, 20, 3, 3, 3, 3, 3, 3, 3, 3, 3])
        self.observation_space = spaces.Box(-high, high, shape=(obs_dim, ))
        # Observations could be what we use as input data to my NN

        # thresholds for done (?)
        self.thresh_stable = .5
        self.thresh_div = 4

        # for making observation:
        self.dataset = WingDataset(0, dt=self.dt, **kwargs)
        self.dataset.set_fixed_mean()

    def done(self):
        # x is greater
        passed = self.state[0] > self.target_point[0]
        # drone unstable
        unstable = np.any(np.absolute(self._state[6:8]) >= self.thresh_stable)
        return unstable or passed

    def prepare_obs(self):
        obs_state, _, obs_ref, _ = self.dataset.prepare_data(
            self.state, self.target_point
        )
        return obs_state, obs_ref

    def state_to_obs(self):
        # get from dataset
        obs_state, obs_ref = self.prepare_obs()
        # concatenate relative position and observation
        obs = torch.cat((obs_ref, obs_state), dim=1)[0].numpy()
        return obs

    def reset(self, x_dist=50, x_std=5, target_point=None):
        if target_point is None:
            rand_y, rand_z = tuple((np.random.rand(2) - .5) * 2 * x_std)
            self.target_point = np.array([x_dist, rand_y, rand_z])
        else:
            self.target_point = np.array(target_point)
        self.zero_reset()
        self.state = self._state
        self.obs = self.state_to_obs()

        self.drone_render_object.set_target([self.target_point])
        return self.obs

    def set_state(self, state):
        self.state = state
        self._state = state

    def get_divergence(self):
        drone_on_line = project_to_line(
            np.zeros(3), self.target_point, self.state[:3]
        )
        div = np.linalg.norm(drone_on_line - self.state[:3])
        return div

    def step(self, action):
        self.state, _ = SimpleWingEnv.step(self, action)
        self.obs = self.state_to_obs()

        div = self.get_divergence()

        done = self.done() or div > self.thresh_div

        if not done:
            reward = self.thresh_div - div
        else:
            reward = 0
        info = {}

        # print()
        # np.set_printoptions(precision=3, suppress=1)
        # print(self.state)
        # print(self.obs)
        # print(div, reward)

        return self.obs, reward, done, info

    def render(self, mode="human"):
        SimpleWingEnv.render(self, mode=mode)


class QuadEnvMario(QuadEnvRL):

    def __init__(self, dynamics, dt, speed_factor=.5, nr_actions=1):
        super().__init__(
            dynamics, dt, speed_factor=speed_factor, nr_actions=nr_actions
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(4, ))

        # state and reference
        self.state_inp_dim = 15
        self.obs_dim = 27
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim, )
        )
