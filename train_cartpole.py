import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from dataset import Dataset
from cartpole_loss import control_loss_function
from evaluate_cartpole import Evaluator
from models.resnet_like_model import Net
from utils.plotting import plot_loss, plot_success
from environments.cartpole_env import construct_states

SAVE_PATH = "trained_models/minimize_x"
NR_EVAL_ITERS = 10

OUT_SIZE = 10  # one action variable between -1 and 1
DIM = 4  # input dimension

net = Net(DIM, OUT_SIZE)
# load network that is trained on standing data
# net = torch.load(
#     os.path.join("models", "minimize_x_brakingWUHU", "model_pendulum")
# )

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

(
    episode_length_mean, episode_length_std, loss_list, pole_angle_mean,
    pole_angle_std, eval_value
) = (list(), list(), list(), list(), list(), list())

NR_EPOCHS = 200
# TRAIN:
for epoch in range(NR_EPOCHS):

    # EVALUATION:
    evaluator = Evaluator()
    # Start in upright position and see how long it is balaned
    success, _ = evaluator.evaluate_in_environment(net, nr_iters=NR_EVAL_ITERS)
    # Start in random position and run 100 times, then get average state
    swing_up_mean, swing_up_std, eval_loss, new_data = evaluator.make_swingup(
        net
    )
    # save and output the evaluation results
    episode_length_mean.append(round(np.mean(success), 3))
    episode_length_std.append(round(np.std(success), 3))
    print(
        "Average episode length: ", episode_length_mean[-1], "std:",
        episode_length_std[-1], "swing up:", swing_up_mean, "std:",
        swing_up_std, "loss:", round(np.mean(eval_loss), 2)
    )
    if swing_up_mean[0] < .3 and swing_up_mean[2] < .2 and np.sum(
        swing_up_mean
    ) < 1 and np.sum(swing_up_std) < 1 and episode_length_mean[-1] > 180:
        print("early stopping")
        break
    eval_value.append(
        swing_up_mean[0] + swing_up_mean[2] +
        (251 - episode_length_mean[-1]) * 0.01
    )
    if epoch > 0 and eval_value[-1] == np.min(eval_value):
        # curr_loss < np.min(loss_list):
        print("New best model")
        torch.save(net, os.path.join(SAVE_PATH, "model_pendulum"))
    print()

    # Renew dataset dynamically
    if epoch % 3 == 0:
        state_data = Dataset(construct_states, num_states=10000)
        if epoch > 5:
            # add the data generated during evaluation
            state_data.add_data(np.array(new_data))
        trainloader = torch.utils.data.DataLoader(
            state_data, batch_size=8, shuffle=True, num_workers=0
        )

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
    except KeyboardInterrupt:
        break

    # Print current loss and possibly save model:
    curr_loss = running_loss / len(state_data)
    print('[%d] loss: %.3f' % (epoch + 1, curr_loss))
    loss_list.append(curr_loss)

# PLOTTING
plot_loss(loss_list, SAVE_PATH)
plot_success(episode_length_mean, episode_length_std, SAVE_PATH)
print('Finished Training')
