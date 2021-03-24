import os
import json
import time
import numpy as np
import torch.optim as optim
import torch
import torch.nn.functional as F

from neural_control.dataset import DroneDataset
from neural_control.drone_loss import (
    drone_loss_function, simply_last_loss, reference_loss, mse_loss,
    weighted_loss
)
from neural_control.environments.drone_dynamics import simple_dynamics_function
from neural_control.environments.flightmare_dynamics import (
    flightmare_dynamics_function
)
from neural_control.controllers.network_wrapper import NetworkWrapper
from evaluate_drone import QuadEvaluator
from neural_control.models.hutter_model import Net
from neural_control.utils.plotting import (
    plot_loss_episode_len, print_state_ref_div
)
try:
    from neural_control.flightmare import FlightmareWrapper
except ModuleNotFoundError:
    pass

DYNAMICS = "simple"
DELTA_T = 0.1
EPOCH_SIZE = 500
SELF_PLAY = 1.5
SELF_PLAY_EVERY_X = 2
PRINT = (EPOCH_SIZE // 30)
NR_EPOCHS = 200
BATCH_SIZE = 8
RESET_STRENGTH = 1.2
MAX_DRONE_DIST = 0.25
THRESH_DIV = .1
THRESH_STABLE = 1.5
USE_MPC_EVERY = 500
NR_EVAL_ITERS = 5
STATE_SIZE = 12
NR_ACTIONS = 10
REF_DIM = 9
ACTION_DIM = 4
LEARNING_RATE = 0.0001
SPEED_FACTOR = .6
MAX_STEPS = int(1000 / int(5 * SPEED_FACTOR))
SAVE = os.path.join("trained_models/drone/test_model")
BASE_MODEL = None #"trained_models/drone/branch_faster_3"
BASE_MODEL_NAME = 'model_quad'

simulate_quadrotor = (
    flightmare_dynamics_function
    if DYNAMICS == "flightmare" else simple_dynamics_function
)

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

# Load model or initialize model
if BASE_MODEL is not None:
    net = torch.load(os.path.join(BASE_MODEL, BASE_MODEL_NAME))
    # load std or other parameters from json
    with open(os.path.join(BASE_MODEL, "param_dict.json"), "r") as outfile:
        param_dict = json.load(outfile)
    STD = np.array(param_dict["std"]).astype(float)
    MEAN = np.array(param_dict["mean"]).astype(float)
else:
    state_data = DroneDataset(
        EPOCH_SIZE,
        SELF_PLAY,
        reset_strength=RESET_STRENGTH,
        max_drone_dist=MAX_DRONE_DIST,
        ref_length=NR_ACTIONS,
        dt=DELTA_T
    )
    in_state_size = state_data.normed_states.size()[1]
    # +9 because adding 12 things but deleting position (3)
    net = Net(
        in_state_size, NR_ACTIONS, REF_DIM, ACTION_DIM * NR_ACTIONS, conv=1
    )
    (STD, MEAN) = (state_data.std, state_data.mean)

# Use cuda if available
global device
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# define optimizer and torch normalization parameters
optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

torch_mean, torch_std = (
    torch.from_numpy(MEAN).float(), torch.from_numpy(STD).float()
)

# save std for normalization during test time
param_dict = {"std": STD.tolist(), "mean": MEAN.tolist()}
# update the used parameters:
param_dict["reset_strength"] = RESET_STRENGTH
param_dict["max_drone_dist"] = MAX_DRONE_DIST
param_dict["horizon"] = NR_ACTIONS
param_dict["ref_length"] = NR_ACTIONS
param_dict["thresh_div"] = THRESH_DIV
param_dict["dt"] = DELTA_T
param_dict["take_every_x"] = SELF_PLAY_EVERY_X
param_dict["thresh_stable"] = THRESH_STABLE
param_dict["use_mpc_every"] = USE_MPC_EVERY
param_dict["dynamics"] = DYNAMICS
param_dict["speed_factor"] = SPEED_FACTOR

with open(os.path.join(SAVE, "param_dict.json"), "w") as outfile:
    json.dump(param_dict, outfile)

# init dataset
state_data = DroneDataset(EPOCH_SIZE, SELF_PLAY, **param_dict)
# Init train loader
trainloader = torch.utils.data.DataLoader(
    state_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

loss_list, success_mean_list, success_std_list = list(), list(), list()

take_every_x = 10
highest_success = 0  #  np.inf
for epoch in range(NR_EPOCHS):

    try:
        # EVALUATE
        print(f"Epoch {epoch} (before)")
        controller = NetworkWrapper(net, state_data, **param_dict)
        eval_env = QuadEvaluator(controller, **param_dict)
        # flightmare
        if DYNAMICS=="real_flightmare":
            eval_env.eval_env = FlightmareWrapper(param_dict["dt"])
        # run with mpc to collect data
        # eval_env.run_mpc_ref("rand", nr_test=5, max_steps=500)
        # run without mpc for evaluation
        suc_mean, suc_std = eval_env.eval_ref(
            "rand", nr_test=10, max_steps=MAX_STEPS, **param_dict
        )

        success_mean_list.append(suc_mean)
        success_std_list.append(suc_std)

        if (epoch + 1) % 3 == 0:
            # renew the sampled data
            state_data.resample_data()
            print(f"Sampled new data ({state_data.num_sampled_states})")
        print(f"self play counter: {state_data.get_eval_index()}")

        if epoch % 5 == 0 and param_dict["thresh_div"] < 2:
            param_dict["thresh_div"] += .05
            print("increased thresh div", param_dict["thresh_div"])

        # save best model
        if epoch > 0 and suc_mean > highest_success:
            highest_success = suc_mean
            print("Best model")
            torch.save(net, os.path.join(SAVE, "model_quad" + str(epoch)))

        print()

        # Training
        tic_epoch = time.time()
        running_loss = 0

        for i, data in enumerate(trainloader, 0):
            # inputs are normalized states, current state is unnormalized in
            # order to correctly apply the action
            in_state, current_state, in_ref_state, ref_states = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # ------------ VERSION 1 (x states at once)-----------------
            actions = net(in_state, in_ref_state)
            actions = torch.sigmoid(actions)
            action_seq = torch.reshape(actions, (-1, NR_ACTIONS, ACTION_DIM))
            # save the reached states
            intermediate_states = torch.zeros(
                in_state.size()[0], NR_ACTIONS, STATE_SIZE
            )
            for k in range(NR_ACTIONS):
                # extract action
                action = action_seq[:, k]
                current_state = simulate_quadrotor(
                    action, current_state, dt=DELTA_T
                )
                intermediate_states[:, k] = current_state

            # print_state_ref_div(
            #     intermediate_states[0].detach().numpy(),
            #     ref_states[0].detach().numpy()
            # )
            # exit()

            loss = simply_last_loss(
                intermediate_states, ref_states[:, -1], action_seq, printout=0
            )

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
