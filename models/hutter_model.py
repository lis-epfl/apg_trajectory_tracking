import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple MLP with three hidden layers, based on RL work of Marco Hutter's
    group
    """

    def __init__(self, in_size, out_size):
        """
        in_size: number of input neurons (features)
        out_size: number of output neurons
        """
        super(Net, self).__init__()
        self.fc_in = nn.Linear(in_size, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc_in(x))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)
        return x
