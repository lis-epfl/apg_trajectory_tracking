import json
import numpy as np
import os
import pandas as pd

from evaluate_drone import QuadEvaluator, load_model
from neural_control.controllers.mpc import MPC
from neural_control.dataset import DroneDataset
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

eval_dict = {"rand": {"nr_test": 50, "max_steps": 1000}}
thresh_stable = 2
thresh_divergence = 3

config = {"render": 0, "dynamics": "flightmare"}

if __name__ == "__main__":
    models_to_evaluate = ["baseline_flightmare", "mpc"]
    names = ["neural_075thrust", "MPC_bl_075thrust"]

    for model_name, save_name in zip(models_to_evaluate, names):
        for use_flightmare in [0]:
            env_name = "flightmare" if use_flightmare else "simple env"
            print(f"---------------- {model_name} in env {env_name} --------")

            out_path = "outputs"
            save_model_name = save_name + f" ({env_name})"
            print("save_model_name", save_model_name)

            df = pd.DataFrame(
                columns=[
                    "Model", "speed_factor", "Trajectory", "Stable steps",
                    "Tracking error", "Speed", "Diverged", "z div"
                ]
            )

            # load model
            model_path = os.path.join("trained_models", "drone", model_name)

            if model_path.split(os.sep)[-1] == "mpc":
                # mpc parameters:
                time_model_params = {
                    "horizon": 30,
                    "dt": .05,
                    "dynamics": "flightmare"
                }
                controller = MPC(**time_model_params)
            # Neural controller
            else:
                controller, time_model_params = load_model(model_path)

            # define evaluation environment
            config.update(time_model_params)

            for speed_factor in [0.6]:

                # define evaluation environment
                evaluator = QuadEvaluator(controller, test_time=1, **config)

                if use_flightmare:
                    evaluator.eval_env = FlightmareWrapper(config["dt"], False)

                for reference, ref_params in eval_dict.items():
                    # run x times
                    for _ in range(ref_params["nr_test"]):

                        # run trajectory tracking
                        (ref_traj, drone_traj,
                         divergence) = evaluator.follow_trajectory(
                             reference,
                             max_nr_steps=ref_params["max_steps"],
                             thresh_stable=thresh_stable,
                             thresh_div=thresh_divergence
                         )
                        # compute results
                        avg_divergence = np.mean(divergence)
                        # speed
                        steps_until_fail = len(drone_traj)
                        if reference == "poly" and len(drone_traj) > 500:
                            drone_traj = drone_traj[100:-500]
                        try:
                            speed_all = evaluator.compute_speed(drone_traj)
                            speed = np.max(speed_all)
                        except ZeroDivisionError:
                            speed = np.nan
                        # divergence in z direction
                        z_div = ref_traj[:, 2] - drone_traj[1:, 2]
                        z_divergence = np.mean(z_div)
                        # did it diverge?
                        was_diverged = int(
                            steps_until_fail < ref_params["max_steps"]
                            and divergence[-1] > thresh_divergence
                        )

                        # log
                        print(
                            reference, "len", steps_until_fail, "div",
                            avg_divergence, "speed", speed, "z div",
                            z_divergence
                        )
                        df.loc[len(df)] = [
                            save_model_name, speed_factor, reference,
                            steps_until_fail, avg_divergence, speed,
                            was_diverged, z_divergence
                        ]

            print(df)
            df.to_csv(os.path.join(out_path, f"eval_{save_model_name}.csv"))
