import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Dataset
from control_loss import control_loss_function
from evaluate import evaluate_in_environment
from environment import CartPoleEnv
from model import Net

NR_EVAL_ITERS = 20

net = Net()

state_data = Dataset(num_states=10000)
trainloader = torch.utils.data.DataLoader(
    state_data, batch_size=8, shuffle=True, num_workers=0
)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

eval_env = CartPoleEnv()
episode_length_mean, episode_length_std, loss_list, pole_angles = (
    list(), list(), list(), list()
)
# TRAIN:
for epoch in range(20):

    # EVALUATION in environment
    success, angles = evaluate_in_environment(
        net, state_data.mean, state_data.std, nr_iters=NR_EVAL_ITERS
    )
    # save and output
    pole_angles.append(np.mean(angles))
    episode_length_mean.append(round(np.mean(success), 3))
    episode_length_std.append(round(np.std(success), 3))
    print(
        "Average episode length: ",
        episode_length_mean[-1], "std:", episode_length_std[-1], "angles:",
        round(pole_angles[-1], 3), "angle std:", round(np.std(angles), 3)
    )

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = control_loss_function(outputs, labels)  # control_loss

        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 300 == 299:  # print every 2000 mini-batches
            print(
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000)
            )
            loss_list.append(running_loss / 2000)
            running_loss = 0.0

# PLOTTING
episode_length_mean = np.array(episode_length_mean)
episode_length_std = np.array(episode_length_std)
plt.figure(figsize=(20, 10))
x = np.arange(len(episode_length_mean))
plt.plot(x, episode_length_mean, '-')
plt.fill_between(
    x,
    episode_length_mean - episode_length_std,
    episode_length_mean + episode_length_std,
    alpha=0.2
)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Average episode length", fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("models/performance.png")

plt.figure(figsize=(15, 8))
plt.plot(loss_list)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.savefig("models/loss.png")

plt.figure(figsize=(15, 8))
plt.plot(pole_angles)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Pole angles", fontsize=18)
plt.savefig("models/pole_angles.png")

# SAVE MODEL
torch.save(net, "models/model_pendulum")
print('Finished Training')
