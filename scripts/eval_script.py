import json
import numpy as np
import os
from evaluate_drone import QuadEvaluator, load_model
from dataset import DroneDataset
import pandas as pd

if __name__ == "__main__":
    model_name = "horizon"
    epoch = ""
    nr_test_straight = 20
    max_steps_straight = 1000
    max_steps_circle = 1000

    df = pd.DataFrame(
        columns=[
            "Horizon", "Trajectory", "Circle plane", "Circle radius",
            "Circle direction", "Number of stable steps", "Tracking error"
        ]
    )

    model_path = os.path.join("trained_models", "drone", model_name)

    net, param_dict = load_model(model_path, epoch=epoch)

    for dt in [0.05, 0.1]:
        print("---------------------------------")
        param_dict["dt"] = dt
        param_dict["max_drone_dist"] = 5 * dt
        horizon = dt * 10

        dataset = DroneDataset(num_states=1, **param_dict)
        evaluator = QuadEvaluator(net, dataset, render=0, **param_dict)

        # STRAIGHT
        for _ in range(nr_test_straight):
            steps_until_fail, avg_divergence = evaluator.straight_traj(
                max_nr_steps=max_steps_straight
            )
            print(steps_until_fail, round(avg_divergence, 2))
            df.loc[len(df)] = [
                horizon, "straight", "0", 0, 0, steps_until_fail,
                avg_divergence
            ]
        # CIRCLE
        for plane in [[0, 1], [0, 2], [1, 2]]:
            for radius in [1, 1.5, 2]:  # np.arange(0.5, 2.5, 0.25):
                for direction in [-1, 1]:
                    (steps_until_fail, avg_divergence) = evaluator.circle_traj(
                        max_nr_steps=max_steps_circle,
                        radius=radius,
                        plane=plane,
                        thresh=1,
                        direction=direction
                    )
                    print(steps_until_fail, avg_divergence)
                    df.loc[len(df)] = [
                        horizon, "circle", plane, radius, direction,
                        steps_until_fail, avg_divergence
                    ]

    print(df)
    df.to_csv("../presentations/evaluate.csv")
