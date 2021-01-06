import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from torch.utils.tensorboard import SummaryWriter

from environments.drone_dynamics import simulate_quadrotor
from environments.drone_env import QuadRotorEnvBase
from drone_loss import drone_loss_function
from utils.plotting import plot_state_variables, plot_position


class DroneModel(nn.Module):

    def __init__(self, horizon=20, state_dim=16):
        super(DroneModel, self).__init__()

        self.horizon = horizon
        self.state_dim = state_dim
        U = [
            nn.Parameter(torch.ones(1, 4) * .4, requires_grad=True)
            for _ in range(horizon)
        ]
        self.U = nn.ParameterList(U)

    def forward(self, current_state):
        intermediate_states = torch.zeros(self.horizon, self.state_dim)
        for i in range(self.horizon):
            # map action to range between 0 and 1
            apply_action = torch.sigmoid(self.U[i])
            current_state = simulate_quadrotor(apply_action, current_state)
            intermediate_states[i] = current_state
        return intermediate_states


def mpc_loss(states, actions, target_states, target_actions):
    return 0


# hyperparameters
nr_epochs = 50
LEARNING_RATE = 0.01
SAVE_PATH = os.path.join("trained_models/mpc/test")
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

writer = SummaryWriter(os.path.join(SAVE_PATH, "runs/experiment1"))

# get data
env = QuadRotorEnvBase()
env.reset()

input_state = torch.from_numpy(np.asarray([env._state.as_np])).float()
print([round(s, 2) for s in input_state.numpy()[0]])

# define target trajectory
# TODO target_states and target_actions

# initialize model and optimizer
drone = DroneModel()
optimizer = optim.SGD(drone.parameters(), lr=LEARNING_RATE, momentum=0.9)
writer.add_graph(drone, input_state)

# optimize
losses, collect_states = [], []
tic = time.time()
for j in range(nr_epochs):
    states = drone(input_state)
    # actions = drone.U
    optimizer.zero_grad()
    # loss
    loss = drone_loss_function(
        torch.unsqueeze(states[-1], 0), start_state=input_state
    )
    # mpc_loss(states, actions, target_states, target_actions)

    # writers and losses
    writer.add_scalar("Loss/train", loss, j)
    for p, param in enumerate(drone.U):
        writer.add_histogram("u_" + str(p), param, j)
    losses.append(loss)
    # if j == 0:
    #     plot_position(
    #         states.detach().numpy()[:, :3],
    #         os.path.join(SAVE_PATH, "states_start.png")
    #     )

    # backprop
    loss.backward()
    optimizer.step()
    # save states
    collect_states.append(states[-1].detach().numpy())

writer.flush()
writer.close()

# output and save
print("time:", time.time() - tic)
# for param in drone.parameters():
#     print(param)
# plot loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.savefig(os.path.join(SAVE_PATH, "loss.png"))
# plot states
collect_states = np.array(collect_states)
print(collect_states.shape)
# states.detach().numpy()[:, :3]
plot_state_variables(
    collect_states.copy(), os.path.join(SAVE_PATH, "states.png")
)
plot_position(collect_states[:, :3], os.path.join(SAVE_PATH, "state_pos.png"))
