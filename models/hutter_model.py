import torch.nn as nn
import torch


class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc1 = nn.Linear(in_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
