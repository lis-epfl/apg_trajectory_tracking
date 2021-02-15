import json
import numpy as np
import os
from evaluate_drone import QuadEvaluator, load_model
from neural_control.dataset import DroneDataset
import pandas as pd

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
thresh_divergence = 1

if __name__ == "__main__":
    model_name = "current_model"
    epoch = ""
    out_path = "../presentations/eval_script_outputs/"

    df = pd.DataFrame(
        columns=[
            "Model", "max_drone_dist", "Trajectory", "Circle plane",
            "Circle radius", "Circle direction", "Stable steps",
            "Tracking error", "Speed"
        ]
    )

    model_path = os.path.join("trained_models", "drone", model_name)

    net, param_dict = load_model(model_path, epoch=epoch)

    dt = 0.05
    for max_drone_dist in [0.25, 0.5]:
        print("---------------------------------")
        param_dict["dt"] = dt
        param_dict["max_drone_dist"] = max_drone_dist
        horizon = 10

        dataset = DroneDataset(1, 1, **param_dict)
        evaluator = QuadEvaluator(
            net, dataset, take_every_x=5000, render=0, **param_dict
        )

        for reference, ref_params in eval_dict.items():
            # run x times
            for _ in range(ref_params["nr_test"]):
                # define circle specifications
                if reference == "circle":
                    circle_args = evaluator.sample_circle()
                else:
                    circle_args = {"plane": 0, "radius": 0, "direction": 0}

                # run trajectory tracking
                _, drone_ref, divergence = evaluator.follow_trajectory(
                    reference,
                    max_nr_steps=ref_params["max_steps"],
                    thresh=thresh_divergence,
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

                # log
                print(
                    reference, "len", steps_until_fail, "div", avg_divergence,
                    "speed", speed
                )
                df.loc[len(df)] = [
                    model_name, max_drone_dist, reference,
                    circle_args["plane"], circle_args["radius"],
                    circle_args["direction"], steps_until_fail, avg_divergence,
                    speed
                ]

    print(df)
    df.to_csv(os.path.join(out_path, f"evaluate_{model_name}.csv"))
