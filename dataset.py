import torch
import numpy as np
import os


def construct_states(
    num_data,
    path_to_states=None,
    save_path="models/minimize_x/state_data.npy"
):
    # define parts of the dataset:
    randomized_runs = .8
    upper_balancing = 1
    one_direction = 1

    # Load precomputed dataset
    if path_to_states is not None:
        state_arr = np.load(path_to_states)
        return state_arr

    # Sample states
    from environment import CartPoleEnv
    env = CartPoleEnv()
    data = []
    # randimized runs
    # while len(data) < num_data * randomized_runs:
    #     # run 100 steps then reset (randomized runs)
    #     for _ in range(10):
    #         action = np.random.rand() - 0.5
    #         state, _, _, _ = env._step(action)
    #         data.append(state)
    #     env._reset()

    # # after randomized runs: run balancing
    while len(data) < num_data:
        fine = False
        # only theta between -0.5 and 0.5
        # env.state = env.state * .5  # TODO
        env.state[2] = (np.random.rand(1) - .5) * .2
        while not fine:
            action = np.random.rand() - 0.5
            state, _, fine, _ = env._step(action)
            data.append(state)
        env._reset()

    # # add one directional steps
    # while len(data) < num_data * one_direction:
    #     action = (-.5) * ((np.random.rand() > .5) * 2 - 1)
    #     for _ in range(30):
    #         state, _, fine, _ = env._step(action)
    #         data.append(state)
    #     env._reset()
    #
    data = np.array(data)

    # # sample only states, no sequences
    # state_limits = np.array([2.4, 5, np.pi, 5])
    # uniform_samples = np.random.rand(num_data, 4) * 2 - 1
    # data = uniform_samples * state_limits

    print("generated random data:", data.shape)
    # eval_data = [data]  # augmentation: , data * (-1)
    # for name in os.listdir("data"):
    #     if name[0] != ".":
    #         eval_data.append(np.load(os.path.join("data", name)))
    # data = np.concatenate(eval_data, axis=0)
    # print("shape after adding evaluation data", data.shape)
    # save data optionally
    if save_path is not None:
        np.save(save_path, data)
    return data[:num_data]


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
    else:
        mean = 0
        std = 1

    states_to_torch = torch.from_numpy(states).float()

    # if we computed mean and std here, return it
    if return_std:
        return states_to_torch, std
    return states_to_torch


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path_to_states=None, num_states=1000):
        # random_positions = np.random.rand(1000, 3) * 10
        state_arr_numpy = construct_states(
            num_states, path_to_states=path_to_states
        )
        state_arr, self.std = raw_states_to_torch(state_arr_numpy)
        self.labels = state_arr
        self.states = state_arr

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.labels[index]
