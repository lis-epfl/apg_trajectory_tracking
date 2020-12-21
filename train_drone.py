import os
import json
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from dataset import Dataset
from drone_loss import drone_loss_function
from environments.drone_dynamics import simulate_quadrotor
from evaluate_drone import QuadEvaluator
from models.hutter_model import Net
from environments.drone_env import construct_states
from utils.plotting import plot_loss, plot_success

EPOCH_SIZE = 5000
USE_NEW_DATA = 500
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 200
BATCH_SIZE = 8
NR_EVAL_ITERS = 30
STATE_SIZE = 16
NR_ACTIONS = 5
ACTION_DIM = 4
SAVE = os.path.join("trained_models/drone/test_model")

net = Net(STATE_SIZE, ACTION_DIM)
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

reference_data = Dataset(
    construct_states, normalize=True, num_states=EPOCH_SIZE
)
(STD, MEAN) = (reference_data.std, reference_data.mean)
torch_mean = torch.from_numpy(MEAN)
torch_std = torch.from_numpy(STD)

# save std for normalization
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)

loss_list, success_mean_list, success_std_list = list(), list(), list()

highest_success = 0
for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    if epoch % 2 == 0:
        state_data = Dataset(
            construct_states,
            normalize=True,
            mean=MEAN,
            std=STD,
            num_states=EPOCH_SIZE,
            # reset_strength=.6 + epoch / 50
        )
        trainloader = torch.utils.data.DataLoader(
            state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

    eval_env = QuadEvaluator(net, MEAN, STD)
    suc_mean, suc_std, pos_responsible, new_data = eval_env.stabilize(
        nr_iters=NR_EVAL_ITERS
    )
    if epoch > 2 and suc_mean > highest_success:
        highest_success = suc_mean
        print("Best model")
        torch.save(net, os.path.join(SAVE, "model_quad" + str(epoch)))

    success_mean_list.append(suc_mean)
    success_std_list.append(suc_std)
    print(f"Epoch {epoch}: Time: {round(suc_mean, 1)} ({round(suc_std, 1)})")
    # print(
    #     "Average position in the end:",
    #     np.mean(np.absolute(new_data), axis=0)[:3]
    # )

    # self-play: add acquired data
    if USE_NEW_DATA > 0 and epoch > 2:
        rand_inds_include = np.random.permutation(len(new_data))[:USE_NEW_DATA]
        state_data.add_data(np.array(new_data)[rand_inds_include])
        print("newly acquired data:", new_data.shape, state_data.states.size())

    running_loss = 0
    try:
        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            inputs, current_state = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # compute loss + backward + optimize
            loss = 0
            for i in range(NR_ACTIONS):
                net_input_state = (current_state - torch_mean) / torch_std
                # forward
                actions = net(net_input_state)
                actions = torch.sigmoid(actions)
                current_state = simulate_quadrotor(actions, current_state)
            loss = drone_loss_function(
                current_state,
                # if the position is responsible more often --> higher weight
                pos_weight=pos_responsible,
                printout=0
            )
            # loss += .1 * i * loss_intermediate
            loss.backward()
            optimizer.step()

            # print statistics
            # print(net.fc3.weight.grad)
            running_loss += loss.item()
            if i % PRINT == PRINT - 1:
                print('Loss: %.3f' % (running_loss / PRINT))
                loss_list.append(running_loss / PRINT)
                running_loss = 0.0
    except KeyboardInterrupt:
        break

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

#
torch.save(net, os.path.join(SAVE, "model_quad"))
plot_loss(loss_list, SAVE)
plot_success(success_mean_list, success_std_list, SAVE)
print("finished and saved.")
