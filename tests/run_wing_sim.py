import torch
import numpy as np

from neural_control.environments.wing_longitudinal_dynamics import long_dynamics
from neural_control.utils.plotting import plot_wing_pos

# # Changes after error correction

state = torch.tensor([[0, 0, 10, 0, 0, 0]]).float()
print("state", state.size())
dt = 1 / 100
t_end = 10.0001
t = np.arange(0, t_end, dt)

## inputs
T = 1.77 / 7  # thrust [N]
del_e = .5
# np.deg2rad(0)  # angle of elevator [rad] (negative is up)

# action
action = torch.tensor([[T, del_e]]).float()  # , [T, del_e]]).float()

state_buff = np.zeros((len(t), len(state[0])))

for i in range(len(t)):
    state_buff[i] = state.numpy()[0]
    state = long_dynamics(state, action, dt)

np.set_printoptions(suppress=True, precision=4)
# print(state_buff)
# np.save("states.npy", state_buff)
# plot_wing_pos(state_buff, "outputs/test_def.jpg")
