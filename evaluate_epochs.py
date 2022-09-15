import argparse
import numpy as np
import pandas as pd
import os
from evaluate_drone import QuadEvaluator, load_model
from neural_control.dynamics.quad_dynamics_flightmare import FlightmareDynamics
from neural_control.environments.drone_env import QuadRotorEnvBase

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model directory as argument")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="current_model",
        help="Directory of model"
    )
    parser.add_argument(
        "-a", "--eval", type=int, default=0, help="run evaluation for steps"
    )
    args = parser.parse_args()

    res_df = []
    for epoch in np.arange(1, 601, 10):
        model_path = os.path.join("trained_models", "quad", args.model)
        controller, params = load_model(model_path, epoch=str(epoch))

        # DYNAMICS &  PARAMETERS
        params["render"] = 0
        params["speed_factor"] = .4
        modified_params = {}
        dynamics = FlightmareDynamics(modified_params=modified_params)
        environment = QuadRotorEnvBase(dynamics, params["dt"])

        # trajectory parameters
        traj_args = {
            "plane": [0, 2],
            "radius": 2,
            "direction": 1,
            "thresh_div": 5,
            "thresh_stable": 2,
            "duration": 10,
            "max_steps": int(1000 / (params["speed_factor"] * 10)) + 1
        }
        # init evaluator
        evaluator = QuadEvaluator(
            controller, environment, test_time=1, **params
        )

        res_dict = evaluator.run_eval(
            "rand", nr_test=args.eval, **traj_args, return_dict=True
        )
        res_dict["epoch"] = epoch
        # print(res_dict)
        res_df.append(res_dict)
    res_df = pd.DataFrame(res_df)
    res_df.set_index("epoch").to_csv("evaluate_epochs.csv")