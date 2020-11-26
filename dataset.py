import torch
import numpy as np


def construct_states(num_data, save_path="data/state_data.npy"):
    from environment import CartPoleEnv
    env = CartPoleEnv()
    data = []
    baseline_episode_length = list()
    while len(data) < num_data:
        is_fine = False
        num_iters = 0
        while not is_fine:
            action = 2 * (np.random.rand() - 0.5)
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


class Dataset(torch.utils.data.Dataset):

    def __init__(self, path_to_states=None, num_states=1000):
        # random_positions = np.random.rand(1000, 3) * 10
        if path_to_states is not None:
            state_arr = np.load(path_to_states)
        else:
            state_arr = construct_states(num_states)
        state_arr = torch.from_numpy(state_arr).float()
        self.labels = state_arr
        self.states = state_arr

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        # Select sample
        return self.states[index], self.labels[index]
