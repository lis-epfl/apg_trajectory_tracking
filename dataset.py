import torch
import numpy as np


def construct_states(
    num_data, path_to_states=None, save_path="data/state_data.npy"
):
    if path_to_states is not None:
        state_arr = np.load(path_to_states)
        return state_arr

    from environment import CartPoleEnv
    env = CartPoleEnv()
    data = []
    baseline_episode_length = list()
    while len(data) < num_data:
        is_fine = False
        num_iters = 0
        while not is_fine:
            action = np.random.rand() - 0.5
            state, _, is_fine, _ = env._step(action)
            # print("action", action, "out:", out)
            data.append(state)
            num_iters += 1
        env._reset()
        baseline_episode_length.append(num_iters)
    data = np.array(data)
    print(
        "generated data:", data.shape, "BASELINE:",
        np.mean(baseline_episode_length), "(std: ",
        np.std(baseline_episode_length), ")"
    )
    if save_path is not None:
        np.save(save_path, data)
    return data[:num_data]


def raw_states_to_torch(states, mean=None, std=None):
    return_mean = mean is None and std is None

    # either input one state at a time (evaluation) or an array
    if len(states.shape) == 1:
        states = np.expand_dims(states, 0)

    if mean is None:
        mean = np.mean(states, axis=0)
    if std is None:
        std = np.std(states, axis=0)

    normed_states = (states - mean) / std
    states_to_torch = torch.from_numpy(normed_states).float()

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