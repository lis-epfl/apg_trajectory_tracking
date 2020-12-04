import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

OUT_SIZE = 10  # one action variable between -1 and 1
DIM = 4  # input dimension


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc1 = nn.Linear(DIM, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, OUT_SIZE)

    def forward(self, x):
        # x = x * torch.from_numpy(np.array([0, 1, 1, 1]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# # RESNET style:
# def __init__(self):
#         super(Net, self).__init__()
#         # conf: in channels, out channels, kernel size
#         self.fc1 = nn.Linear(DIM, 80)
#         self.fc2 = nn.Linear(80, 80)
#         self.fc25 = nn.Linear(80, 40)
#         self.fc3 = nn.Linear(40, OUT_SIZE)

#     def forward(self, x):
#         # x = x * torch.from_numpy(np.array([0, 1, 1, 1]))
#         x = F.relu(self.fc1(x))
#         shortcut = x
#         x = F.relu(self.fc2(x)) + shortcut
#         x = F.relu(self.fc25(x))
#         x = self.fc3(x)
#         return x