import torch
import numpy as np
import os
from environments.cartpole_env import construct_states


def raw_states_to_torch(states, normalize=False, std=None):
    """
    Helper function to convert numpy state array to normalized tensors
    Argument states:
            One state (list of length 4) or array with x states (x times 4)
    """
    return_std = std is None

    # either input one state at a time (evaluation) or an array
    if len(states.shape) == 1:
        states = np.expand_dims(states, 0)

    # save mean and std column wise
    if normalize:
        # can't use mean!
        if std is None:
            std = np.std(states, axis=0)
        states = states / std
        assert np.all(np.isclose(np.std(states, axis=0), 1))
    else:
        std = 1

    states_to_torch = torch.from_numpy(states).float()

    # if we computed mean and std here, return it
    if return_std:
        return states_to_torch, std
    return states_to_torch


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        state_sampling_method,
        path_to_states=None,
        num_states=1000,
        normalize=False
    ):
        # random_positions = np.random.rand(1000, 3) * 10
        state_arr_numpy = state_sampling_method(num_states)
        state_arr, self.std = raw_states_to_torch(
            state_arr_numpy, normalize=normalize
        )
        self.labels = state_arr
        self.states = state_arr

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.labels[index]
