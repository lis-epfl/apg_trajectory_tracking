import numpy as np
import torch.optim as optim
import torch

from dataset import Dataset
from drone_loss import control_loss
# from evaluate_drone import Evaluator # TODO
from models.resnet_like_model import Net
from environments.drone_env import construct_states

net = Net(21, 4)
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

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        lam = epoch / NR_EPOCHS
        loss = control_loss_function(outputs, labels, lambda_factor=lam)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 300 == 299:  # print every 2000 mini-batches
            # loss = control_loss_function(outputs, labels, printout=True)
            # print()
            print(
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000)
            )
            loss_list.append(running_loss / 2000)
            running_loss = 0.0
