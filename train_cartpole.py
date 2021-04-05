import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from neural_control.dataset import CartpoleDataset
from neural_control.cartpole_loss import control_loss_function
from evaluate_cartpole import Evaluator
from neural_control.models.resnet_like_model import Net
from neural_control.plotting import plot_loss, plot_success
from neural_control.environments.cartpole_env import construct_states

SAVE_PATH = "trained_models/cartpole/current_model"
NR_EVAL_ITERS = 10
NR_SWINGUP_ITERS = 20
USE_NEW_DATA = 1000

OUT_SIZE = 10
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
        net, nr_iters=NR_SWINGUP_ITERS
    )
    # save and output the evaluation results
    episode_length_mean.append(round(np.mean(success), 3))
    episode_length_std.append(round(np.std(success), 3))
    print(
        "Average episode length: ", episode_length_mean[-1], "std:",
        episode_length_std[-1], "swing up:", swing_up_mean, "std:",
        swing_up_std, "loss:", round(np.mean(eval_loss), 2)
    )
    if swing_up_mean[0] < .5 and swing_up_mean[2] < .5 and np.sum(
        swing_up_mean
    ) < 3 and np.sum(swing_up_std) < 1 and episode_length_mean[-1] > 180:
        print("early stopping")
        break
    performance_swingup = swing_up_mean[0] + swing_up_mean[
        2] + (251 - episode_length_mean[-1]) * 0.01
    if epoch > 0 and performance_swingup < np.min(eval_value):
        # curr_loss < np.min(loss_list):
        print("New best model")
        torch.save(net, os.path.join(SAVE_PATH, "model_pendulum" + str(epoch)))
    print()
    eval_value.append(performance_swingup)

    # Renew dataset dynamically
    if epoch % 3 == 0:
        state_data = CartpoleDataset(num_states=10000)
        if epoch > 5:
            # add the data generated during evaluation
            rand_inds_include = np.random.permutation(len(new_data)
                                                      )[:USE_NEW_DATA]
            state_data.add_data(np.array(new_data)[rand_inds_include])
        trainloader = torch.utils.data.DataLoader(
            state_data, batch_size=8, shuffle=True, num_workers=0
        )
        print(f"------- new dataset {len(state_data)}---------")

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
            loss = control_loss_function(
                outputs, labels, lambda_factor=lam, printout=0
            )
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

torch.save(net, os.path.join(SAVE_PATH, "model_pendulum"))

# PLOTTING
plot_loss(loss_list, SAVE_PATH)
plot_success(episode_length_mean, episode_length_std, SAVE_PATH)
print('Finished Training')
