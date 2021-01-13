import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple MLP with three hidden layers, based on RL work of Marco Hutter's
    group
    """

    def __init__(self, state_dim, ref_dim, out_size):
        """
        in_size: number of input neurons (features)
        out_size: number of output neurons
        """
        super(Net, self).__init__()
        self.states_in = nn.Linear(state_dim, 64)
        # todo: maybe conv2d
        self.ref_in = nn.Linear(ref_dim, 64)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, out_size)

    def forward(self, state, ref):
        # process state and reference differently
        state = torch.tanh(self.states_in(state))
        ref = torch.tanh(self.ref_in(ref))
        # concatenate
        x = torch.hstack((state, ref))
        # normal feed-forward
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)
        return x
