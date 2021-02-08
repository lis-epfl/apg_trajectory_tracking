from ruamel.yaml import YAML, dump, RoundTripDumper
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import os
import numpy as np
import torch

from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper

from evaluate_drone import load_model, QuadEvaluator
from neural_control.environments.drone_env import QuadRotorEnvBase
from neural_control.environments.drone_dynamics import action_to_rotor
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

        def obs_to_np_state(self, obs):
                # obs is position, euler, velocity, and body rates (w)
                transformed_state = np.zeros(16)
                # add pos, att, vel
                transformed_state[:9] = obs[0, :9].copy()
                transformed_state[3:6] *=.1
                # add rotor speeds
                transformed_state[9:13] = self._state.rotor_speeds
                # add body rates
                transformed_state[13:] = obs[0, 9:]
                return transformed_state

        def rotor_to_force(self, rotor):
                a = 1.3298253500372892e-06
                b = 0.0038360810526746033
                c = -1.7689986848125325
                return a * rotor**2 + b * rotor + c

        def action_to_fm(self, action):
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

        def reset(self, strength=.8):
                """
                Interface to flightmare reset
                """ 
                super().reset()
                print(self._state.as_np[:6])
                obs = self.env.reset()
                print(obs[:6])
                # convert obs from flightmare to state here
                state = self.obs_to_np_state(obs)
                print(state)
                # set own state (current_np_state)
                self._state.from_np(state)
                return self._state

        def zero_reset(self, position_x=0, position_y=0, position_z=2):
                """
                TODO: should ideally set vel to zero
                and position to arguments pos_x etc
                """
                return self.reset()

        def step(self, action, thresh=.4):
                """
                Overwrite step methods of drone_env
                Use dynamics model implementde in flightmare instead
                """
                # TODO: convert action from model to flightmare input
                np.set_printoptions(suppress=True, precision=2)
                # print("state before", self._state.as_np)
                # print("raw action", action)
                action = self.action_to_fm(action)
                # TODO: how to input dt into fm env?--> sim_dt_ variable
                obs, rew, done, infos = self.env.step(action)
                # print("obs", obs)
                # print("rew", rew) # TODO: how is reward computed?
                # TODO: convert obs to numpy state as in my mode
                state = self.obs_to_np_state(obs)
                self._state.from_np(state)
                # print(self._state.as_np)

                # check whether it is still stable
                stable = True # np.all(np.absolute(state[3:5]) < thresh)
                return state, stable

if __name__=="__main__":
        model_name = "current_model"
        epoch = ""

        # load model
        model_path = os.path.join("trained_models", "drone", model_name)
        net, param_dict = load_model(model_path, epoch=epoch)

        # load config
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                        "/flightlib/configs/vec_env.yaml", 'r'))
        # print(dump(cfg, Dumper=RoundTripDumper))
        cfg["env"]["num_envs"] = 1

        # initialize dataset
        dataset = DroneDataset(num_states=1, **param_dict)

        # make evaluator
        evaluator = QuadEvaluator(net, dataset, render=1, **param_dict)
        evaluator.eval_env = FlightmareWrapper(
                param_dict["dt"],
                wrapper.FlightEnvVec(QuadrotorEnv_v1(
                dump(cfg, Dumper=RoundTripDumper), False))
        )

        # print(evaluator.eval_env.rotor_to_force(np.zeros(4)+400))
        # exit()
        reference_traj, drone_traj = evaluator.follow_trajectory(
            'straight', max_nr_steps=100
        )
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

# class FlightmareEvaluator(QuadEvaluator):
#         def __init__(
#                 self,
#                 cfg,
#                 model,
#                 dataset,
#                 horizon=5,
#                 max_drone_dist=0.1,
#                 render=0,
#                 dt=0.02,
#                 **kwargs
#         ):
#                 super(self).__init__(self,
#                         model,
#                         dataset,
#                         horizon=5,
#                         max_drone_dist=0.1,
#                         render=0,
#                         dt=0.02,
#                         **kwargs
#                 )
#                 # initialize flighmare environment
#                 self.eval_env = wrapper.FlightEnvVec(QuadrotorEnv_v1(
#                         dump(cfg, Dumper=RoundTripDumper), False))
#                 self.render = 0

        # def predict_actions(self, current_np_state, ref_states)
        # probably don't even need to overwrite this
