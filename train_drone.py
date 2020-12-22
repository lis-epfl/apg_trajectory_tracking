import os
import json
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from dataset import Dataset
from drone_loss import drone_loss_function, trajectory_loss
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
NR_EVAL_ITERS = 5
STATE_SIZE = 16
NR_ACTIONS = 5
ACTION_DIM = 4
LEARNING_RATE = 0.001
SAVE = os.path.join("trained_models/drone/test_model")

net = Net(STATE_SIZE, ACTION_DIM)  # NR_ACTIONS *
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

reference_data = Dataset(
    construct_states, normalize=True, num_states=EPOCH_SIZE
)
(STD, MEAN) = (reference_data.std, reference_data.mean)
torch_mean, torch_std = torch.from_numpy(MEAN), torch.from_numpy(STD)

# save std for normalization
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)

loss_list, success_mean_list, success_std_list = list(), list(), list()

target_state = torch.zeros(STATE_SIZE)
target_state[2] = 2
mask = torch.ones(STATE_SIZE)
mask[9:13] = 0  # rotor speeds don't matter
loss_weights = mask  # torch.tensor([]) TODO
target_state = ((target_state - torch_mean) / torch_std) * mask


def adjust_learning_rate(optimizer, epoch, every_x=5):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = LEARNING_RATE * (0.1**(epoch // every_x))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    print()
    print(f"Epoch {epoch-1}")
    eval_env = QuadEvaluator(net, MEAN, STD)
    suc_mean, suc_std, new_data = eval_env.evaluate(
        nr_hover_iters=NR_EVAL_ITERS, nr_traj_iters=NR_EVAL_ITERS
    )
    success_mean_list.append(suc_mean)
    success_std_list.append(suc_std)
    if epoch > 0:
        if suc_mean > highest_success:
            highest_success = suc_mean
            print("Best model")
            torch.save(net, os.path.join(SAVE, "model_quad" + str(epoch)))
        print("Loss:", round(running_loss / i, 2))

    # self-play: add acquired data
    if USE_NEW_DATA > 0 and epoch > 2 and len(new_data) > 0:
        rand_inds_include = np.random.permutation(len(new_data))[:USE_NEW_DATA]
        state_data.add_data(np.array(new_data)[rand_inds_include])
        # print("new added data:", new_data.shape, state_data.states.size())

    running_loss = 0
    try:
        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            inputs, current_state = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # # TRAJECTORY LOSS:
            # actions = net(inputs)
            # actions = torch.sigmoid(actions)

            # # unnormalized state of the drone after the action
            # # drone_state = simulate_quadrotor(actions, current_state)
            # action_seq = torch.reshape(actions, (-1, NR_ACTIONS, ACTION_DIM))
            # for act_ind in range(action_seq.size()[1]):
            #     action = action_seq[:, act_ind, :]
            #     current_state = simulate_quadrotor(action, current_state)
            # # normalize
            # drone_state = (current_state - torch_mean) / torch_std
            # pout = 1 if False else 0
            # loss_traj = trajectory_loss(
            #     inputs, target_state, drone_state, mask=mask, printout=pout
            # )
            # loss = torch.sum(loss_traj)
            # # # reshape to get sequence of actions

            # compute loss + backward + optimize
            # actions = net(inputs)
            # actions = torch.sigmoid(actions)
            # action_seq = torch.reshape(actions, (-1, NR_ACTIONS, ACTION_DIM))
            for k in range(NR_ACTIONS):
                # action = action_seq[:, k]
                # VERSION 2: predict one action at a time
                net_input_state = (current_state - torch_mean) / torch_std
                action = net(net_input_state)
                action = torch.sigmoid(action)
                current_state = simulate_quadrotor(action, current_state)
            # Only compute loss after last action
            loss = drone_loss_function(
                current_state,
                # if the position is responsible more often --> higher weight
                pos_weight=0,
                printout=0
            )
            # loss += .1 * i * loss_intermediate
            loss.backward()
            optimizer.step()

            # print statistics
            # print(net.fc3.weight.grad)
            running_loss += loss.item()

        loss_list.append(running_loss / i)
    except KeyboardInterrupt:
        break

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

#
torch.save(net, os.path.join(SAVE, "model_quad"))
plot_loss(loss_list, SAVE)
plot_success(success_mean_list, success_std_list, SAVE)
print("finished and saved.")
