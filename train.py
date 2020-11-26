import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import Dataset, raw_states_to_torch
from control_loss import ControlLoss
from environment import CartPoleEnv

OUT_SIZE = 1  # one action variable between -1 and 1
DIM = 4  # input dimension
NR_EVAL_ITERS = 10


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # conf: in channels, out channels, kernel size
        self.fc1 = nn.Linear(DIM, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, OUT_SIZE)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # TODO: okay to do this in torch? or need to return logits


net = Net()

state_data = Dataset(num_states=10000)
trainloader = torch.utils.data.DataLoader(
    state_data, batch_size=8, shuffle=True, num_workers=0
)

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
loss_fun = ControlLoss()

eval_env = CartPoleEnv()
episode_length_mean, episode_length_std, loss_list = list(), list(), list()
# TRAIN:
for epoch in range(5):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = loss_fun(outputs, labels)  # control_loss
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 300 == 299:  # print every 2000 mini-batches
            print(
                '[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000)
            )
            loss_list.append(running_loss)
            running_loss = 0.0

    # evaluation:
    with torch.no_grad():
        success = np.zeros(NR_EVAL_ITERS)
        for it in range(NR_EVAL_ITERS):
            is_fine = False
            episode_length_counter = 0
            new_state = eval_env.state
            while not is_fine:
                torch_state = raw_states_to_torch(
                    new_state, mean=state_data.mean, std=state_data.std
                )
                # torch_state = torch.from_numpy(np.expand_dims(new_state,
                #                                               0)).float()
                action = torch.sigmoid(net(torch_state)).item() - .5
                # 2 * (np.random.rand() - 0.5)
                new_state, _, is_fine, _ = eval_env._step(action)
                # print(torch_state.numpy(), new_state, action)
                # print("action", action, "out:", out)
                episode_length_counter += 1
            success[it] = episode_length_counter
            eval_env._reset()
        # save and output
        episode_length_mean.append(round(np.mean(success), 3))
        episode_length_std.append(round(np.std(success), 3))
        print(
            "Average episode length: ", episode_length_mean[-1], "std:",
            episode_length_std[-1]
        )

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
plt.xlabel("Epoch")
plt.ylabel("Average episode length")
plt.savefig("models/performance.png")

plt.figure(figsize=(20, 10))
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("models/loss.png")

torch.save(net, "models/model_pendulum")
print('Finished Training')
