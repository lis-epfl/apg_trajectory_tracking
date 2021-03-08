import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from neural_control.dataset import WingDataset
from neural_control.drone_loss import trajectory_loss
from neural_control.environments.wing_longitudinal_dynamics import long_dynamics
from neural_control.models.hutter_model import Net
from evaluate_fixed_wing import FixedWingEvaluator
from neural_control.controllers.network_wrapper import FixedWingNetWrapper
from neural_control.utils.plotting import plot_loss_episode_len

DELTA_T = 0.01
EPOCH_SIZE = 3000
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 200
BATCH_SIZE = 8
STATE_SIZE = 6
NR_ACTIONS = 20
REF_DIM = 2
ACTION_DIM = 2
LEARNING_RATE = 0.001
SAVE = os.path.join("trained_models/wing/test_model")
BASE_MODEL = None  # "trained_models/wing/half_corrected_working"
BASE_MODEL_NAME = 'model_wing'

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

# Load model or initialize model
if BASE_MODEL is not None:
    net = torch.load(os.path.join(BASE_MODEL, BASE_MODEL_NAME))
    # load std or other parameters from json
    with open(os.path.join(BASE_MODEL, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)
else:
    param_dict = {"dt": DELTA_T, "horizon": NR_ACTIONS}
    net = Net(
        STATE_SIZE - REF_DIM, 1, REF_DIM, ACTION_DIM * NR_ACTIONS, conv=False
    )

# Use cuda if available
global device
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# define optimizer and torch normalization parameters
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

# init dataset
state_data = WingDataset(EPOCH_SIZE, **param_dict)
param_dict = state_data.get_means_stds(param_dict)

with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)

# Init train loader
trainloader = torch.utils.data.DataLoader(
    state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

loss_list, success_mean_list, success_std_list = list(), list(), list()

take_steps = 1
highest_success = np.inf
for epoch in range(NR_EPOCHS):

    try:
        # EVALUATE
        print(f"Epoch {epoch} (before)")
        controller = FixedWingNetWrapper(net, state_data, **param_dict)
        eval_env = FixedWingEvaluator(controller, **param_dict)

        nr_test = 20 if epoch == 0 else 10
        suc_mean, suc_std = eval_env.run_eval(nr_test=nr_test)
        success_mean_list.append(suc_mean)
        success_std_list.append(suc_std)

        if (epoch + 1) % 4 == 0:
            # renew the sampled data
            state_data.resample_data()
            print(f"Sampled new data ({state_data.num_sampled_states})")

        # save best model
        if epoch > 0 and suc_mean < highest_success:
            highest_success = suc_mean
            print("Best model")
            torch.save(net, os.path.join(SAVE, "model_wing" + str(epoch)))

        print()

        # Training
        tic_epoch = time.time()
        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, current_state, in_ref_state, ref_state = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # ------------ VERSION 1 (x states at once)-----------------
            actions = net(in_state, in_ref_state)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(actions, (-1, NR_ACTIONS, ACTION_DIM))
            # unnnormalize state
            # start_state = current_state.clone()
            intermediate_states = torch.zeros(
                in_state.size()[0], NR_ACTIONS, STATE_SIZE
            )
            drone_state = current_state
            for k in range(NR_ACTIONS):
                # extract action
                action = action_seq[:, k]
                drone_state = long_dynamics(drone_state, action, dt=DELTA_T)
                intermediate_states[:, k] = drone_state

            loss = trajectory_loss(
                current_state, ref_state, intermediate_states, printout=0
            )

            # Backprop
            loss.backward()
            # print(net.fc_out.weight.grad.size(), net.fc_out.weight.grad)
            optimizer.step()

            running_loss += loss.item()

        loss_list.append(running_loss / i)

    except KeyboardInterrupt:
        break
    print("Loss:", round(running_loss / i, 2))
    # print("time one epoch", time.time() - tic_epoch)

# Save model
torch.save(net, os.path.join(SAVE, "model_wing"))
plot_loss_episode_len(
    success_mean_list,
    success_std_list,
    loss_list,
    save_path=os.path.join(SAVE, "performance.png")
)
print("finished and saved.")
