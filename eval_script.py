import json
import numpy as np
import os
from evaluate_drone import QuadEvaluator, load_model
from dataset import DroneDataset
import pandas as pd

if __name__ == "__main__":
    model_name = "horizon"
    epoch = ""
    nr_test_straight = 3
    max_steps_straight = 200
    max_steps_circle = 200

    df = pd.DataFrame(
        columns=[
            "dt", "traj", "plane", "radius", "direction", "steps_stable",
            "avg_div"
        ]
    )

    model_path = os.path.join("trained_models", "drone", model_name)

    net, param_dict = load_model(model_path, epoch=epoch)

    dt = 0.05
    max_drone_dist = 0.25
    param_dict["dt"] = dt
    param_dict["max_drone_dist"] = max_drone_dist

    dataset = DroneDataset(num_states=1, **param_dict)
    evaluator = QuadEvaluator(net, dataset, render=0, **param_dict)

    # STRAIGHT
    for _ in range(nr_test_straight):
        steps_until_fail, avg_divergence = evaluator.straight_traj(
            max_nr_steps=max_steps_straight
        )
        print(steps_until_fail, avg_divergence)
        df.loc[len(df)] = [
            dt, "straight", "", 0, 0, steps_until_fail, avg_divergence
        ]

    # CIRCLE
    for plane in [[0, 1], [0, 2], [1, 2]]:
        for radius in [1, 1.5]:  # np.arange(0.5, 2.5, 0.25):
            for direction in [-1, 1]:
                steps_until_fail, avg_divergence = evaluator.circle_traj(
                    max_nr_steps=max_steps_circle,
                    radius=radius,
                    plane=plane,
                    thresh=1,
                    direction=direction
                )
                print(steps_until_fail, avg_divergence)
                df.loc[len(df)] = [
                    dt, "circle", plane, radius, direction, steps_until_fail,
                    avg_divergence
                ]
    print(df)
    df.to_csv("../presentations/evaluate.csv")
