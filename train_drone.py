import numpy as np
import torch.optim as optim
import torch

from dataset import Dataset
from drone_loss import drone_loss_function
# from evaluate_drone import Evaluator # TODO
from models.resnet_like_model import Net
from environments.drone_env import construct_states

net = Net(20, 4)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

EPOCH_SIZE = 1000
NR_EPOCHS = 2
running_loss = 0
loss_list = list()

for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    state_data = Dataset(
        construct_states, normalize=True, num_states=EPOCH_SIZE
    )
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=8, shuffle=True, num_workers=0
    )

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, current_state = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        actions = net(inputs)
        actions = torch.sigmoid(actions)
        lam = epoch / NR_EPOCHS
        loss = drone_loss_function(current_state, actions)
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 30 == 29:  # print every 2000 mini-batches
            # loss = control_loss_function(outputs, labels, printout=True)
            # print()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
            loss_list.append(running_loss)
            running_loss = 0.0
