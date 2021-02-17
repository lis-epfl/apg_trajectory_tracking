import numpy as np

# PREDEINE SOME TRAJECTORIES

collected_trajectories = {
    "eight": np.array([
                [0, 0, 0],
                [-2, 2, 1],
                [0, 4, 2],
                [2, 2, 3],
                [0, 0, 4],
                [-2, -2, 3],
                [0, -4, 2],
                [2, -2, 1],
                [0, 0, 0]
            ]) * 2,
    "curve": np.array(
            [
                [-1.5, 0, 2], 
                [-1, 1, 1], 
                [-.5, -1, 2], 
                [0, -3, 3],
                [1, -2, 5],
                [2, -1, 4],
                [3, 1, 3]
            ]
        )
}