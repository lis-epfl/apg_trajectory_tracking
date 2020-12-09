import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import Dataset
from cartpole_loss import control_loss_function
from evaluate_cartpole import Evaluator
from models.resnet_like_model import Net
from environments.cartpole_env import construct_states

NR_EVAL_ITERS = 10

net = Net()
# load network that is trained on standing data
# net = torch.load(
#     os.path.join("models", "minimize_x_brakingWUHU", "model_pendulum")
# )

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

(
    episode_length_mean, episode_length_std, loss_list, pole_angle_mean,
    pole_angle_std
) = (list(), list(), list(), list(), list())

NR_EPOCHS = 200
# TRAIN:
for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    state_data = Dataset(
        construct_states, num_states=10000
    )  # 1 epoch is 10000
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=8, shuffle=True, num_workers=0
    )

    # EVALUATION:
    evaluator = Evaluator()
    # Start in upright position and see how long it is balaned
    success, _ = evaluator.evaluate_in_environment(net, nr_iters=NR_EVAL_ITERS)
    # Start in random position and run 100 times, then get average state
    swing_up_mean, swing_up_std, eval_loss = evaluator.make_swingup(net)
    # save and output the evaluation results
    episode_length_mean.append(round(np.mean(success), 3))
    episode_length_std.append(round(np.std(success), 3))
    print()
    print(
        "Average episode length: ", episode_length_mean[-1], "std:",
        episode_length_std[-1], "swing up:", swing_up_mean, "std:",
        swing_up_std, "loss:", round(np.mean(eval_loss), 2)
    )
    if swing_up_mean[0] < .3 and swing_up_mean[2] < .2 and np.sum(
        swing_up_mean
    ) < 1 and np.sum(swing_up_std) < 1 and episode_length_mean[-1] > 200:
        print("early stopping")
        break

    try:
        running_loss = 0.0
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
    except KeyboardInterrupt:
        break

# PLOTTING
SAVE_PATH = "trained_models/minimize_x"
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
plt.savefig(os.path.join(SAVE_PATH, "performance.png"))

plt.figure(figsize=(15, 8))
plt.plot(loss_list)
plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Loss", fontsize=18)
plt.savefig(os.path.join(SAVE_PATH, "loss.png"))

# pole_angle_mean = np.array(pole_angle_mean)
# pole_angle_std = np.array(pole_angle_std)
# plt.figure(figsize=(20, 10))
# plt.plot(x, pole_angle_mean, '-')
# plt.fill_between(
#     x,
#     pole_angle_mean - pole_angle_std,
#     pole_angle_mean + pole_angle_std,
#     alpha=0.2
# )
# plt.xlabel("Epoch", fontsize=18)
# plt.xticks(fontsize=18)
# plt.yticks(fontsize=18)
# plt.ylabel("Pole angles", fontsize=18)
# plt.savefig("models/pole_angles.png")

# SAVE MODEL
torch.save(net, os.path.join(SAVE_PATH, "model_pendulum"))
print('Finished Training')
