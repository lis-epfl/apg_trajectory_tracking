from ruamel.yaml import YAML, dump, RoundTripDumper
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
import numpy as np
import torch
import argparse

from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper

from evaluate_drone import load_model, QuadEvaluator
from neural_control.environments.drone_env import QuadRotorEnvBase
# from neural_control.environments.drone_dynamics import action_to_rotor
from neural_control.dataset import DroneDataset
from neural_control.utils.plotting import (
    plot_state_variables, plot_trajectory, plot_position, plot_suc_by_dist,
    plot_drone_ref_coords
)

class FlightmareWrapper(QuadRotorEnvBase):
        """
        Wrapper around the wrapper around the quadrotor environment
        in flightmare
        Necessary to use the evaluator of the pytorch environment
        """
        def __init__(self, dt, flightmare_env):
                super().__init__(dt)
                self.env = flightmare_env

        def transform_borders(self, x, switch_sign=0):
                new = np.sign(x) * min([abs(x), (3.14 - abs(x))])
                if new!=x and switch_sign:
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
                transformed_state[4] = self.transform_borders(obs[0,4])
                transformed_state[5] = self.transform_borders(obs[0, 3])
                # add body rates
                transformed_state[9:] = obs[0, 9:]
                return transformed_state

        def rotor_to_force(self, rotor):
                a = 1.3298253500372892e-06
                b = 0.0038360810526746033
                c = -1.7689986848125325
                return a * rotor**2 + b * rotor + c

        def action_to_fm_ROTOR(self, action):
                """
                Map from action in my model to an input to flighmare model
                """
                # Transform action to rotor speeds
                torch_action = torch.from_numpy(np.expand_dims(action,0))
                torch_rotor = torch.from_numpy(np.expand_dims(self._state.rotor_speeds,0))
                rotor_speeds = action_to_rotor(torch_action, torch_rotor).numpy()
                # update state
                self._state._rotorspeeds = rotor_speeds[0]
                # transform rotor speeds to force
                force = self.rotor_to_force(rotor_speeds)
                return force.astype(np.float32)

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
                action[0, 1:]*=1.5
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
                if not stable:
                        np.set_printoptions(precision=3, suppress=True)
                        print("unstable!")
                        print("obs", obs)
                        print("state", state)
                return state, stable

if __name__=="__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument(
                "-m",
                "--model",
                type=str,
                default="current_model",
                help="Directory of model"
        )
        parser.add_argument(
                "-e",
                "--epoch",
                type=str,
                default="",
                help="Which epoch of the model"
        )
        parser.add_argument(
                "-u",
                "--unity",
                type=int,
                default=0,
                help="1 if unity rendering"
        )
        parser.add_argument(
                "-r",
                "--ref",
                default="straight",
                type=str,
                help="reference trajectory type"
        )
        args = parser.parse_args()

        render = 1
        if args.unity:
                render = 0

        model_name = args.model
        epoch = args.epoch

        # load model
        model_path = os.path.join("trained_models", "drone", model_name)
        net, param_dict = load_model(model_path, epoch=epoch)

        # load config
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                        "/flightlib/configs/vec_env.yaml", 'r'))
        # print(dump(cfg, Dumper=RoundTripDumper))
        cfg["env"]["num_envs"] = 1
        if args.unity:
                cfg["env"]["render"] = "yes"

        # initialize dataset
        dataset = DroneDataset(1, 1, **param_dict)

        # make evaluator
        evaluator = QuadEvaluator(net, dataset, render=render, **param_dict)
        evaluator.eval_env = FlightmareWrapper(
                param_dict["dt"],
                wrapper.FlightEnvVec(QuadrotorEnv_v1(
                dump(cfg, Dumper=RoundTripDumper), False))
        )

        if args.unity:
                evaluator.eval_env.env.connectUnity()

        # print(evaluator.eval_env.rotor_to_force(np.zeros(4)+400))
        # exit()
        reference_traj, drone_traj, div = evaluator.follow_trajectory(
            args.ref, max_nr_steps=1000, radius=3, plane=[0,1], thresh=.9
        )
        if args.unity:
                evaluator.eval_env.env.disconnectUnity()

        print(len(reference_traj))
        plot_drone_ref_coords(
            drone_traj[1:, :3], reference_traj,
            os.path.join(model_path, "fm_coords.png")
        )
        plot_state_variables(drone_traj,
            os.path.join(model_path, "fm_all.png") )
        # # From previous flightmare tests:
        # obs, done, ep_len = (env.reset(), False, 0)
        # print(env._extraInfoNames)
        # print("obs prev:", obs)
        # act = np.random.rand(1,4).astype(np.float32)
        # obs, rew, done, infos = env.step(act)
        # print("obs", obs)

