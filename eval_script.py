import json
import numpy as np
import os
import pandas as pd
from ruamel.yaml import YAML, dump, RoundTripDumper
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)


from evaluate_drone import QuadEvaluator, load_model
from neural_control.dataset import DroneDataset
from flightgym import QuadrotorEnv_v1
from rpg_baselines.envs import vec_env_wrapper as wrapper
from test_flightmare import FlightmareWrapper

eval_dict = {
    "straight": {
        "nr_test": 20,
        "max_steps": 1000
    },
    "circle": {
        "nr_test": 20,
        "max_steps": 1000
    },
    "poly": {
        "nr_test": 20,
        "max_steps": 1000
    }
}
thresh_stable = 1
thresh_divergence = 3

if __name__ == "__main__":
    model_name = "current_model"
    epoch = ""
    out_path = "outputs"
    use_flightmare = True

    df = pd.DataFrame(
        columns=[
            "Model", "max_drone_dist", "Trajectory", "Circle plane",
            "Circle radius", "Circle direction", "Stable steps",
            "Tracking error", "Speed", "Diverged"
        ]
    )

    model_path = os.path.join("trained_models", "drone", model_name)

    net, param_dict = load_model(model_path, epoch=epoch)

    if use_flightmare:
         # load config
        cfg = YAML().load(open(os.environ["FLIGHTMARE_PATH"] +
                        "/flightlib/configs/vec_env.yaml", 'r'))
        cfg["env"]["num_envs"] = 1

    dt = 0.05
    for max_drone_dist in [0.25, 0.5]:
        print("---------------------------------")
        param_dict["dt"] = dt
        param_dict["max_drone_dist"] = max_drone_dist
        horizon = 10

        dataset = DroneDataset(1, 1, **param_dict)
        evaluator = QuadEvaluator(
            net,
            dataset,
            take_every_x=5000,
            render=0,
            **param_dict
        )
        if use_flightmare:
            evaluator.eval_env = FlightmareWrapper(
                    param_dict["dt"],
                    wrapper.FlightEnvVec(QuadrotorEnv_v1(
                    dump(cfg, Dumper=RoundTripDumper), False))
            )

        for reference, ref_params in eval_dict.items():
            # run x times
            for _ in range(ref_params["nr_test"]):
                # define circle specifications
                if reference == "circle":
                    circle_args = evaluator.sample_circle()
                    # if use_flightmare:
                    #     circle_args["plane"] = [0,1]
                else:
                    circle_args = {"plane": 0, "radius": 0, "direction": 0}

                # run trajectory tracking
                _, drone_ref, divergence = evaluator.follow_trajectory(
                    reference,
                    max_nr_steps=ref_params["max_steps"],
                    thresh_stable=thresh_stable,
                    thresh_div=thresh_divergence,
                    **circle_args
                )
                # compute results
                avg_divergence = np.mean(divergence)
                steps_until_fail = len(drone_ref)
                if reference == "poly" and len(drone_ref) > 500:
                    drone_ref = drone_ref[100:-500]
                try:
                    speed = evaluator.compute_speed(drone_ref)
                except ZeroDivisionError:
                    speed = np.nan

                was_diverged = int(steps_until_fail < ref_params["max_steps"]
                        and divergence[-1] > thresh_divergence)

                # log
                print(
                    reference, "len", steps_until_fail, "div", avg_divergence,
                    "speed", speed, "diverged?", was_diverged
                )
                df.loc[len(df)] = [
                    model_name, max_drone_dist, reference,
                    circle_args["plane"], circle_args["radius"],
                    circle_args["direction"], steps_until_fail, avg_divergence,
                    speed, was_diverged
                ]

    print(df)
    df.to_csv(os.path.join(out_path, f"evaluate_flightmare_{model_name}.csv"))
