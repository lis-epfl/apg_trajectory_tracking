import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, in_size, out_size):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc_in = nn.Linear(in_size, 100)
        self.fc1 = nn.Linear(100, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 100)
        self.fc8 = nn.Linear(100, 100)
        self.fc_last = nn.Linear(100, 40)
        self.fc_out = nn.Linear(40, out_size)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        # 1st resnet block
        shortcut = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) + shortcut
        # 2nd resnet block
        shortcut = x
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x)) + shortcut
        # 3rd resnet block
        shortcut = x
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x)) + shortcut
        # 4th block
        shortcut = x
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x)) + shortcut
        # output layers
        x = F.relu(self.fc_last(x))
        x = self.fc_out(x)
        return x
