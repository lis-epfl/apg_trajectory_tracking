import torch
import numpy as np
from neural_control.plotting import print_state_ref_div
from neural_control.dynamics.cartpole_dynamics import CartpoleDynamics
device = "cpu"
torch.autograd.set_detect_anomaly(True)
zero_tensor = torch.zeros(3).to(device)

rates_prior = torch.tensor([.5, .5, .5])


def quad_mpc_loss(states, ref_states, action_seq, printout=0):
    # MATCH TO MPC
    # self._Q_u = np.diag([50, 1, 1, 1])
    # self._Q_pen = np.diag([100, 100, 100, 10, 10, 10, 10, 10, 10, 1, 1, 1])
    pos_factor = 10
    u_thrust_factor = 5
    u_rates_factor = 0.1
    av_factor = 0.1
    vel_factor = 1

    position_loss = torch.sum((states[:, :, :3] - ref_states[:, :, :3])**2)
    velocity_loss = torch.sum((states[:, :, 6:9] - ref_states[:, :, 6:9])**2)

    av_loss = torch.sum(states[:, :, 9:12]**2)
    u_thrust_loss = torch.sum((action_seq[:, :, 0] - .5)**2)
    u_rates_loss = torch.sum((action_seq[:, :, 1:] - rates_prior)**2)

    loss = (
        pos_factor * position_loss + vel_factor * velocity_loss +
        av_factor * av_loss + u_rates_factor * u_rates_loss +
        u_thrust_factor * u_thrust_loss
    )

    if printout:
        print_state_ref_div(
            states[0].detach().numpy(), ref_states[0].detach().numpy()
        )
    return loss


def quad_loss_last(states, last_ref_state, action_seq, printout=0):
    angvel_factor = 2e-2
    vel_factor = 0.1
    pos_factor = 10
    yaw_factor = 10
    action_factor = .1

    action_loss = torch.sum((action_seq[:, :, 0] - .5)**2)

    position_loss = torch.sum((states[:, -1, :3] - last_ref_state[:, :3])**2)
    velocity_loss = torch.sum((states[:, -1, 6:9] - last_ref_state[:, 6:9])**2)

    ang_vel_error = torch.sum(states[:, :, 9:11]**2
                              ) + yaw_factor * torch.sum(states[:, :, 11]**2)
    # TODO: do on all intermediate states again?

    loss = (
        angvel_factor * ang_vel_error + pos_factor * position_loss +
        vel_factor * velocity_loss + action_factor * action_loss
    )
    if printout:
        print_state_ref_div(
            states[0].detach().numpy(), last_ref_state[0].detach().numpy()
        )
    return loss


action_prior = torch.tensor([.5, .5, .5])


def fixed_wing_mpc_loss(drone_states, linear_reference, action, printout=0):
    # Use costs from MPC:
    # self._Q_u = np.diag([0, 10, 10, 10])
    # self._Q_pen = np.diag([1000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    action_factor = 0.1
    pos_factor = 10

    action_loss = torch.sum((action[:, :, 1:] - action_prior)**2)
    pos_loss = torch.sum((drone_states[:, :, :3] - linear_reference)**2)
    loss = pos_factor * pos_loss + action_factor * action_loss
    return loss


def fixed_wing_last_loss(drone_states, linear_reference, action, printout=0):
    # action_loss = torch.sum((action[:, :, 1:] - action_prior)**2)
    loss = torch.sum((drone_states[:, :3] - linear_reference)**2)
    # av_loss = 0.1 * torch.sum(drone_states[:, 1:, 9:]**2)
    # att_loss = torch.sum(drone_states[:, 6:8]**2)
    # loss = pos_loss  #  + att_loss

    if printout:
        import numpy as np
        print(linear_reference.size(), drone_states.size())
        np.set_printoptions(precision=3, suppress=True)
        print("target")
        print(linear_reference.detach())
        print("drone states")
        print(drone_states.detach())
        print("action")
        print(action.detach().numpy()[0])
    return loss


def cartpole_loss(action, state, lambda_factor=.4, printout=0):

    # bring action into -1 1 range
    action = torch.sigmoid(action) - .5

    horizon = action.size()[1]

    # update state iteratively for each proposed action
    for i in range(horizon):
        state = simulate_cartpole(state, action[:, i])
    abs_state = torch.abs(state)

    pos_loss = state[:, 0]**2
    # velocity losss is low when x is high
    vel_loss = abs_state[:, 1] * (2.4 - abs_state[:, 0])**2
    angle_loss = 3 * abs_state[:, 2]
    # high angle velocity is fine if angle itself is high
    angle_vel_loss = .1 * abs_state[:, 3] * (torch.pi - abs_state[:, 2])**2
    loss = .1 * (pos_loss + vel_loss + angle_loss + angle_vel_loss)

    if printout:
        print("position_loss", pos_loss[0].item())
        print("vel_loss", vel_loss[0].item())
        # print("factor", factor[0].item())
        print("angle loss", angle_loss[0].item())
        print("angle vel", angle_vel_loss[0].item())
        print()
    # print(fail)
    return torch.sum(loss)  # + angle_acc)


mpc_losses = torch.tensor([0, 3, 10, 1])


def cartpole_loss_mpc(states, ref_states, actions):
    loss = (states - ref_states)**2 * mpc_losses
    loss_actions = torch.sum(actions**2)
    # angle_loss = (states[:, :, 2] - ref_states[:, :, 2])**2
    # angle_vel_loss = (states[:, :, 3] - ref_states[:, :, 3])**2
    # loss = 10 * angle_loss + angle_vel_loss
    return torch.sum(loss) + 0.01 * loss_actions


def cartpole_loss_balance(state):
    abs_state = torch.abs(state)
    angle_loss = 3 * abs_state[:, 2]
    # high angle velocity is fine if angle itself is high
    angle_vel_loss = .1 * abs_state[:, 3] * (torch.pi - abs_state[:, 2])**2
    loss = .1 * (angle_loss + angle_vel_loss)
    return torch.sum(loss)


def cartpole_loss_swingup(state, lambda_factor=.4, printout=0):

    abs_state = torch.abs(state)

    pos_loss = state[:, 0]**2
    # velocity losss is low when x is high
    vel_loss = abs_state[:, 1] * (2.4 - abs_state[:, 0])**2
    angle_loss = 3 * abs_state[:, 2]
    # high angle velocity is fine if angle itself is high
    angle_vel_loss = .1 * abs_state[:, 3] * (torch.pi - abs_state[:, 2])**2
    loss = .1 * (pos_loss + vel_loss + angle_loss + angle_vel_loss)

    if printout:
        print("position_loss", pos_loss[0].item())
        print("vel_loss", vel_loss[0].item())
        # print("factor", factor[0].item())
        print("angle loss", angle_loss[0].item())
        print("angle vel", angle_vel_loss[0].item())
        print()
    # print(fail)
    return torch.sum(loss)  # + angle_acc)
