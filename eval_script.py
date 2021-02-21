import json
import numpy as np
import os
import pandas as pd

from evaluate_drone import QuadEvaluator, load_model
from neural_control.dataset import DroneDataset
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

eval_dict = {
    "straight": {
        "nr_test": 0,
        "max_steps": 1000
    },
    "circle": {
        "nr_test": 0,
        "max_steps": 1000
    },
    "poly": {
        "nr_test": 50,
        "max_steps": 1000
    }
}
thresh_stable = 1.5
thresh_divergence = 3

if __name__ == "__main__":
    models_to_evaluate = ["current_model", "horizon_1", "motion_prior"]
    names = ["normal", "larger step size", "motion prior"]

    for model_name, save_name in zip(models_to_evaluate, names):
        for use_flightmare in [0,1]:
            env_name = "flightmare" if use_flightmare else "simple env"
            print(f"---------------- {model_name} in env {env_name} --------")

            # model_name = "current_model"
            epoch = ""
            out_path = "outputs"
            save_model_name = save_name + f" ({env_name})"
            print("save_model_name", save_model_name)
            # use_flightmare = True

            df = pd.DataFrame(
                columns=[
                    "Model", "max_drone_dist", "Trajectory", "Circle plane",
                    "Circle radius", "Circle direction", "Stable steps",
                    "Tracking error", "Speed", "Diverged"
                ]
            )

            model_path = os.path.join("trained_models", "drone", model_name)

            net, param_dict = load_model(model_path, epoch=epoch)

            for speed in [0.5, 1]:
                max_drone_dist = speed * param_dict["dt"] * param_dict["horizon"]
                param_dict["max_drone_dist"] = max_drone_dist

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
                            param_dict["dt"], False
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
                            save_model_name, max_drone_dist, reference,
                            circle_args["plane"], circle_args["radius"],
                            circle_args["direction"], steps_until_fail, avg_divergence,
                            speed, was_diverged
                        ]

            print(df)
            df.to_csv(os.path.join(out_path, f"eval_{save_model_name}.csv"))
