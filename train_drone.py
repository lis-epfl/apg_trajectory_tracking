import numpy as np
import torch.optim as optim
import torch

from dataset import Dataset
from drone_loss import control_loss
# from evaluate_drone import Evaluator # TODO
from models.resnet_like_model import Net
from environments.drone_env import construct_states

net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

NR_EPOCHS = 2

for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    state_data = Dataset(
        construct_states, num_states=10000
    )  # 1 epoch is 10000
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=8, shuffle=True, num_workers=0
    )
