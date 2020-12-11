import torch
from environments.drone_dynamics import simulate_quadrotor


def attitude_loss(state):
    """
    Compute loss to static position
    """
    # weighting
    angle_factor = 1.0
    angvel_factor = 1e-2

    angle_error = torch.sum(state[:, 3:6]**2, axis=1)
    ang_vel_error = torch.sum(state[:, 17:20]**2, axis=1)
    return (angle_factor * angle_error) + (angvel_factor * ang_vel_error)


def drone_loss_function(current_state, action_seq, printout=0):
    """
    Computes loss for applying an action to the current state by comparing to
    the target state
    Arguments:
        current_state: array with x entries describing attitude and velocity
        action: control signal of dimension 4 (thrust of rotors)
    """
    for act_ind in range(action_seq.size()[1]):
        action = action_seq[:, act_ind, :]
        current_state = simulate_quadrotor(action, current_state, dt=0.02)
    # add attitude loss to loss for wrong position
    position_loss = (current_state[:, 2] - 1)**2
    loss = attitude_loss(current_state) + 500 * position_loss
    # print("loss", loss)
    if printout:
        print()
        print("attitude loss", attitude_loss(current_state)[0])
        print("position loss", position_loss[0])
    return torch.sum(loss)