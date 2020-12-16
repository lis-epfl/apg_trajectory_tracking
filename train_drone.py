import os
import json
import torch.optim as optim
import torch

from dataset import Dataset
from drone_loss import drone_loss_function, trajectory_loss
from environments.drone_dynamics import simulate_quadrotor
from evaluate_drone import QuadEvaluator
from models.hutter_model import Net
from environments.drone_env import construct_states
from utils.plotting import plot_loss, plot_success

EPOCH_SIZE = 10000
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 200
BATCH_SIZE = 8
NR_EVAL_ITERS = 30
NR_ACTIONS = 1
ACTION_DIM = 4
STATE_SIZE = 20
LEARNING_RATE = 0.01
SAVE = os.path.join("trained_models/drone/test_model")

net = Net(STATE_SIZE, NR_ACTIONS * ACTION_DIM)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)  #, momentum=0.9)

reference_data = Dataset(
    construct_states, normalize=True, num_states=EPOCH_SIZE
)
(STD, MEAN) = (reference_data.std, reference_data.mean)
torch_mean, torch_std = torch.from_numpy(MEAN), torch.from_numpy(STD)

loss_list, success_mean_list, success_std_list = list(), list(), list()

target_state = torch.zeros(STATE_SIZE)
target_state[2] = 2
mask = torch.ones(STATE_SIZE)
mask[6:17] = 0  # rotor speeds don't matter, but want to optimize position,
# attitude, angular vel etc
# normalize the state
target_state = ((target_state - torch_mean) / torch_std) * mask


def adjust_learning_rate(optimizer, epoch, every_x=5):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = LEARNING_RATE * (0.1**(epoch // every_x))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


highest_success = 0
for epoch in range(NR_EPOCHS):

    # Generate data dynamically
    state_data = Dataset(
        construct_states,
        normalize=True,
        mean=MEAN,
        std=STD,
        num_states=EPOCH_SIZE
    )
    trainloader = torch.utils.data.DataLoader(
        state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    eval_env = QuadEvaluator(net, MEAN, STD)
    suc_mean, suc_std, pos_responsible = eval_env.stabilize(
        nr_iters=NR_EVAL_ITERS
    )
    if suc_mean > highest_success:
        highest_success = suc_mean
        print("Best model")
        torch.save(net, os.path.join(SAVE, "model_quad" + str(epoch)))

    success_mean_list.append(suc_mean)
    success_std_list.append(suc_std)
    print(f"Epoch {epoch}: Time: {round(suc_mean, 1)} ({round(suc_std, 1)})")

    running_loss = 0
    try:
        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            inputs, current_state = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            actions = net(inputs)
            actions = torch.sigmoid(actions)

            # unnormalized state of the drone after the action
            drone_state = simulate_quadrotor(actions, current_state)
            # normalize
            drone_state = (drone_state - torch_mean) / torch_std
            pout = 1 if False else 0
            loss_traj = trajectory_loss(
                inputs, target_state, drone_state, mask=mask, printout=pout
            )
            loss = torch.sum(loss_traj)
            # # reshape to get sequence of actions

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
# save std for normalization
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)
#
torch.save(net, os.path.join(SAVE, "model_quad"))
plot_loss(loss_list, SAVE)
plot_success(success_mean_list, success_std_list, SAVE)
print("finished and saved.")
