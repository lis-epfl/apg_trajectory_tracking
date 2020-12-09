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
    return angle_factor * angle_error + angvel_factor * ang_vel_error
    # (pitch**2 + roll**2 + yaw**2)


def control_loss(current_state, action):
    """
    Computes loss for applying an action to the current state by comparing to
    the target state
    Arguments:
        current_state: array with x entries describing attitude and velocity
        action: control signal of dimension 4 (thrust of rotors)
    """
    resulting_state = simulate_quadrotor(action, current_state, dt=0.02)
    loss = attitude_loss(resulting_state)
    return torch.sum(loss)
