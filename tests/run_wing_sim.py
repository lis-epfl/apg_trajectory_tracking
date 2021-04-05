import torch
import numpy as np

from neural_control.dynamics.fixed_wing_dynamics import FixedWingDynamics
from neural_control.plotting import plot_wing_pos

dyn = FixedWingDynamics()

# # Changes after error correction

state = torch.zeros(1, 12).float()
state[0, 3] = 11.5
print("state", state.size())
dt = 1 / 100
t_end = 10.0001
t = np.arange(0, t_end, dt)

## inputs
T = 1.9 / 7  # thrust [N]
del_e = 0.5
del_a = 0.5
del_r = 35 / 40
# np.deg2rad(0)  # angle of elevator [rad] (negative is up)

# action
action = torch.tensor([[T, del_e, del_a,
                        del_r]]).float()  # , [T, del_e]]).float()

print(state.size())

state_buff = np.zeros((len(t), len(state[0])))

for i in range(len(t)):
    state_buff[i] = state.numpy()[0]
    state = dyn.simulate_fixed_wing(state, action, dt)

np.set_printoptions(suppress=True, precision=4)
print(state_buff)
# np.save("states.npy", state_buff)
# plot_wing_pos(state_buff, "outputs/test_def.jpg")
