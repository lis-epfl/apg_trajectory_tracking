import torch
import numpy as np

from neural_control.environments.wing_longitudinal_dynamics import long_dynamics
from neural_control.utils.plotting import plot_wing_pos

state = torch.tensor([[0, 0, 10, 0, 0, 0], [0, 0, 10, 0, 0, 0]]).float()
print("state", state.size())
dt = 1 / 100
t_end = 10
t = np.arange(0, t_end, dt)

## inputs
T = 1.3  # thrust [N]
del_e = np.deg2rad(-1)  # angle of elevator [rad] (negative is up)

# action
action = torch.tensor([[T, del_e], [T, del_e]]).float()
print("action", action.size())

state_buff = np.zeros((len(t), len(state[0])))

for i in range(int(t_end / dt)):
    state_buff[i] = state.numpy()[1]
    state = long_dynamics(state, action, dt)

np.set_printoptions(suppress=True, precision=4)
print(state_buff)
plot_wing_pos(state_buff, "outputs/test_def.jpg")
