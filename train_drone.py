import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from dataset import DroneDataset
from drone_loss import drone_loss_function, trajectory_loss, reference_loss
from environments.drone_dynamics import simulate_quadrotor
from evaluate_drone import QuadEvaluator
from models.hutter_model import Net
from utils.plotting import plot_loss_episode_len

EPOCH_SIZE = 5000
USE_NEW_DATA = 0  # 250
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 200
BATCH_SIZE = 8
RESET_STRENGTH = 1.2
MAX_DRONE_DIST = 0.2
THRESH_DIV = .2
NR_EVAL_ITERS = 5
STATE_SIZE = 13
NR_ACTIONS = 5
REF_DIM = 9
ACTION_DIM = 4
LEARNING_RATE = 0.001
SAVE = os.path.join("trained_models/drone/test_model")
BASE_MODEL = os.path.join("trained_models/drone/ref_good_selfplay_straight")
BASE_MODEL_NAME = 'model_quad'

# Load model or initialize model
if BASE_MODEL is not None:
    net = torch.load(os.path.join(BASE_MODEL, BASE_MODEL_NAME))
    # load std or other parameters from json
    with open(os.path.join(BASE_MODEL, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)
    STD = np.array(param_dict["std"]).astype(float)
    MEAN = np.array(param_dict["mean"]).astype(float)
    state_data = DroneDataset(
        mean=MEAN,
        std=STD,
        num_states=EPOCH_SIZE,
        reset_strength=RESET_STRENGTH,
        max_drone_dist=MAX_DRONE_DIST,
        ref_length=NR_ACTIONS
    )
else:
    state_data = DroneDataset(
        num_states=EPOCH_SIZE,
        reset_strength=RESET_STRENGTH,
        max_drone_dist=MAX_DRONE_DIST,
        ref_length=NR_ACTIONS
    )
    net = Net(STATE_SIZE, NR_ACTIONS, REF_DIM, ACTION_DIM * NR_ACTIONS)
    (STD, MEAN) = (state_data.std, state_data.mean)

# Use cuda if available
global device
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# Initialize train loader
trainloader = torch.utils.data.DataLoader(
    state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

# define optimizer and torch normalization parameters
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

torch_mean, torch_std = (
    torch.from_numpy(MEAN).float(), torch.from_numpy(STD).float()
)

# save std for normalization during test time
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
# update the used parameters:
param_dict["reset"] = RESET_STRENGTH
param_dict["max_drone_dist"] = MAX_DRONE_DIST
param_dict["horizon"] = NR_ACTIONS
param_dict["treshold_divergence"] = THRESH_DIV

with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)

loss_list, success_mean_list, success_std_list = list(), list(), list()

# target_state = torch.zeros(STATE_SIZE)
# mask = torch.ones(STATE_SIZE)
# mask[6:13] = 0  # rotor speeds and xyz velocity don't matter
# loss_weights = mask.clone()
# target_state = ((target_state - torch_mean) / torch_std) * mask


def adjust_learning_rate(optimizer, epoch, every_x=5):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = LEARNING_RATE * (0.1**(epoch // every_x))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


highest_success = 0  # np.inf
for epoch in range(NR_EPOCHS):

    try:
        # Generate data dynamically
        state_data.sample_data(self_play=0)

        print(f"Epoch {epoch} (before)")
        eval_env = QuadEvaluator(net, state_data, **param_dict)
        suc_mean, suc_std = eval_env.eval_ref()

        success_mean_list.append(suc_mean)
        success_std_list.append(suc_std)

        # save best model
        if epoch > 0 and suc_mean > highest_success:
            highest_success = suc_mean
            print("Best model")
            torch.save(net, os.path.join(SAVE, "model_quad" + str(epoch)))

        # self play
        print(
            f"Self play data: {round(100*state_data.eval_counter/EPOCH_SIZE)}%"
        )

        print()

        # Training
        tic_epoch = time.time()
        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, ref_world, ref_body = data
            # unnormalize TODO: maybe return from dataset simply
            current_state = in_state * torch_std + torch_mean
            current_state[:, :3] = 0

            # TODO: Could input :3 to NN with vel (problem: normalization)

            # zero the parameter gradients
            optimizer.zero_grad()

            # ------------ VERSION 1 (x states at once)-----------------
            actions = net(in_state[:, 3:], ref_body)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(actions, (-1, NR_ACTIONS, ACTION_DIM))
            # unnnormalize state
            # start_state = current_state.clone()
            intermediate_states = torch.zeros(
                in_state.size()[0], NR_ACTIONS, STATE_SIZE + 3
            )
            for k in range(NR_ACTIONS):
                # normalize loss by the start distance
                action = action_seq[:, k]
                # ----------- VERSION 2: predict one action at a time --------
                # start_dist = torch.sum(current_state[:, :3]**2, axis=1)
                # net_input_state = (current_state - torch_mean) / torch_std
                # action = net(net_input_state)
                # action = torch.sigmoid(action)
                current_state = simulate_quadrotor(action, current_state)
                intermediate_states[:, k] = current_state  # [:, :3]

                # Only compute loss after last action
                # 1) --------- drone loss function --------------
            loss = reference_loss(intermediate_states, ref_world, printout=0)
            # ------------- VERSION 3: Trajectory loss -------------
            # drone_state = (current_state - torch_mean) / torch_std
            # loss = trajectory_loss(
            #     inputs,
            #     target_state,
            #     drone_state,
            #     loss_weights=loss_weights,
            #     mask=mask,
            #     printout=0
            # )

            # Backprop
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        loss_list.append(running_loss / i)

    except KeyboardInterrupt:
        break
    print("Loss:", round(running_loss / i, 2))
    # print("time one epoch", time.time() - tic_epoch)

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

# Save model
torch.save(net, os.path.join(SAVE, "model_quad"))
plot_loss_episode_len(
    success_mean_list,
    success_std_list,
    loss_list,
    save_path=os.path.join(SAVE, "performance.png")
)
print("finished and saved.")
