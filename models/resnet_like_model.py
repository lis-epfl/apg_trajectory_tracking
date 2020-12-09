import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc1 = nn.Linear(in_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc25 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # 1st resnet block
        shortcut = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x)) + shortcut
        # 2nd resnet block
        shortcut = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x)) + shortcut
        # 3rd resnet block
        shortcut = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x)) + shortcut
        # 4th block
        shortcut = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x)) + shortcut
        x = F.relu(self.fc25(x))
        x = self.fc3(x)
        return x
