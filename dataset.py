import torch
import numpy as np


def construct_states(
    num_data, path_to_states=None, save_path="models/state_data.npy"
):
    # Load precomputed dataset
    if path_to_states is not None:
        state_arr = np.load(path_to_states)
        return state_arr

    # Sample states
    from environment import CartPoleEnv
    env = CartPoleEnv()
    data = []
    baseline_episode_length = list()
    while len(data) < num_data:
        num_iters = 0
        if len(data) > 0.8 * num_data:
            env.state = (np.random.rand(4) - .5) * .1
        # run 100 steps then reset
        for _ in range(100):
            action = np.random.rand() - 0.5
            state, _, _, _ = env._step(action)
            data.append(state)
            num_iters += 1
        env._reset()
        baseline_episode_length.append(num_iters)
    data = np.array(data)
    print(
        "generated data:",
        data.shape,
        # "BASELINE:",
        # np.mean(baseline_episode_length), "(std: ",
        # np.std(baseline_episode_length), ")"
    )
    # save data optionally
    if save_path is not None:
        np.save(save_path, data)
    return data[:num_data]


def raw_states_to_torch(states, normalize=False, mean=None, std=None):
    """
    Helper function to convert numpy state array to normalized tensors
    Argument states: 
            One state (list of length 4) or array with x states (x times 4)
    """
    return_mean = mean is None and std is None

    # either input one state at a time (evaluation) or an array
    if len(states.shape) == 1:
        states = np.expand_dims(states, 0)

    # save mean and std column wise
    if normalize:
        if mean is None:
            mean = np.mean(states, axis=0)
        if std is None:
            std = np.std(states, axis=0)
        states = (states - mean) / std
    else:
        mean = 0
        std = 1

    states_to_torch = torch.from_numpy(states).float()

    # if we computed mean and std here, return it
    if return_mean:
        return states_to_torch, mean, std
    return states_to_torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path_to_states=None, num_states=1000):
        # random_positions = np.random.rand(1000, 3) * 10
        state_arr_numpy = construct_states(
            num_states, path_to_states=path_to_states
        )
        state_arr, self.mean, self.std = raw_states_to_torch(state_arr_numpy)
        self.labels = state_arr
        self.states = state_arr

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.labels[index]
