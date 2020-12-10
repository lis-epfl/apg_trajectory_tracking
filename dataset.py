import torch
import numpy as np
import os
from environments.cartpole_env import construct_states


def raw_states_to_torch(
    states, normalize=False, std=None, mean=None, return_std=False
):
    """
    Helper function to convert numpy state array to normalized tensors
    Argument states:
            One state (list of length 4) or array with x states (x times 4)
    """
    # either input one state at a time (evaluation) or an array
    if len(states.shape) == 1:
        states = np.expand_dims(states, 0)

    # save mean and std column wise
    if normalize:
        # can't use mean!
        if std is None:
            std = np.std(states, axis=0)
        if mean is None:
            mean = np.mean(states, axis=0)
        states = (states - mean) / std
        # assert np.all(np.isclose(np.std(states, axis=0), 1))
    else:
        std = 1

    # np.save("data_backup/quad_data.npy", states)

    states_to_torch = torch.from_numpy(states).float()

    # if we computed mean and std here, return it
    if return_std:
        return states_to_torch, mean, std
    return states_to_torch


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        state_sampling_method,
        num_states=1000,
        normalize=False,
        mean=None,
        std=None
    ):
        self.std = std
        self.mean = mean
        # sample states
        state_arr_numpy = state_sampling_method(num_states)
        # convert to normalized tensors
        state_arr, self.mean, self.std = raw_states_to_torch(
            state_arr_numpy, normalize=normalize, std=std, return_std=True
        )
        self.labels = state_arr
        self.states = state_arr

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.labels[index]
