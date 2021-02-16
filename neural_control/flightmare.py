from ruamel.yaml import YAML, dump, RoundTripDumper
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
import numpy as np
import torch

from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper

from neural_control.environments.drone_env import QuadRotorEnvBase


class FlightmareWrapper(QuadRotorEnvBase):
    """
        Wrapper around the wrapper around the quadrotor environment
        in flightmare
        Necessary to use the evaluator of the pytorch environment
        """

    def __init__(self, dt, unity_render=False):
        super().__init__(dt)
        # load config
        cfg = YAML().load(
            open(
                os.environ["FLIGHTMARE_PATH"] +
                "/flightlib/configs/vec_env.yaml", 'r'
            )
        )
        cfg["env"]["num_envs"] = 1
        # set up rendering
        if unity_render:
            cfg["env"]["render"] = "yes"
        self.env = wrapper.FlightEnvVec(
            QuadrotorEnv_v1(dump(cfg, Dumper=RoundTripDumper), False)
        )

    def transform_borders(self, x, switch_sign=0):
        new = np.sign(x) * min([abs(x), (3.14 - abs(x))])
        if new != x and switch_sign:
            new = -1 * new
        return new

    def obs_to_np_state(self, obs):
        # obs is position, euler, velocity, and body rates (w)
        transformed_state = np.zeros(12)
        # add pos
        transformed_state[:3] = obs[0, :3].copy()
        # add vel
        transformed_state[6:9] = obs[0, 6:9].copy()
        # attitude --> zyx to xyz and remove discontinouities
        transformed_state[3] = self.transform_borders(obs[0, 5], switch_sign=1)
        transformed_state[4] = self.transform_borders(obs[0, 4])
        transformed_state[5] = self.transform_borders(obs[0, 3])
        # add body rates
        transformed_state[9:] = obs[0, 9:]
        return transformed_state

    def action_to_fm(self, action):
        # action is normalized between 0 and 1 --> rescale
        act_fm = action.copy()
        # total_thrust
        act_fm[0] = action[0] * 10 - 5 + 7
        # ang momentum
        act_fm[1:] = action[1:] - .5
        return np.expand_dims(act_fm, 0).astype(np.float32)

    def reset(self, strength=.8):
        """
                Interface to flightmare reset
                """
        super().reset()
        obs = self.env.reset()
        self.raw_obs = obs
        # convert obs from flightmare to state here
        state = self.obs_to_np_state(obs)
        # set own state (current_np_state)
        self._state.from_np(state)
        return self._state

    def zero_reset(self, position_x=0, position_y=0, position_z=2):
        """
                TODO: should ideally set vel to zero
                and position to arguments pos_x etc
                """
        return self.reset()

    def step(self, action, thresh=.8):
        """
                Overwrite step methods of drone_env
                Use dynamics model implementde in flightmare instead
                """
        # TODO: convert action from model to flightmare input
        np.set_printoptions(suppress=True, precision=2)
        # print("state before", self._state.as_np)
        # print("obs before", self.raw_obs)
        # print("raw action", action)
        action = self.action_to_fm(action)
        action[0, 1:] *= 1.5
        # print("action", action)
        # action = np.random.rand(*action.shape).astype(np.float32)
        # action = np.ones(action.shape).astype(np.float32) * 2.5 + action
        # print(action)
        # np.zeros(action.shape).astype(np.float32) # TODO
        # TODO: how to input dt into fm env?--> sim_dt_ variable
        obs, rew, done, infos = self.env.step(action)
        self.raw_obs = obs
        # print("obs after", obs)
        # print("rew", rew) # TODO: how is reward computed?
        # TODO: convert obs to numpy state as in my mode
        state = self.obs_to_np_state(obs)
        self._state.from_np(state)
        # if state[8]>2:
        #         print("action", action)
        #         print("state", self._state.as_np)
        # exit()
        # check whether it is still stable
        stable = np.all(np.absolute(state[3:5]) < thresh)
        return state, stable
