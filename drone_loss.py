import torch
from environments.drone_dynamics import simulate_quadrotor


def attitude_loss(state):
    """
    Compute loss to static position
    """
    # weighting
    angle_factor = 1.0
    angvel_factor = 0.05

    angle_error = torch.sum(state[:, 3:6]**2, axis=1)
    ang_vel_error = torch.sum(state[:, 17:20]**2, axis=1)
    return (angle_factor * angle_error) + (angvel_factor * ang_vel_error)


def drone_loss_function(current_state, action_seq, printout=0, pos_weight=1):
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
    position_loss = (current_state[:, 2] - 2)**2
    loss = attitude_loss(current_state) + (1 + pos_weight * 20) * position_loss
    # print("loss", loss)
    if printout:
        print()
        print("attitude loss", attitude_loss(current_state)[0])
        print("position loss", position_loss[0])
    return torch.sum(loss)


def project_to_line(a_on_line, b_on_line, p):
    ap = torch.unsqueeze(p - a_on_line, 2)
    ab = b_on_line - a_on_line
    # normalize
    norm = torch.sum(ab**2, axis=1)
    v = torch.unsqueeze(ab, 2)
    # vvT * (p-a)
    dot = torch.matmul(v, torch.transpose(v, 1, 2))
    product = torch.squeeze(torch.matmul(dot, ap))
    # add a to move away from origin again
    projected = a_on_line + (product.t() / norm).t()

    return projected


def trajectory_loss(state, target_state, drone_state, mask=None, printout=0):
    """
    Loss for attemtping to traverse from state to target_state but ending up
    at drone_state
    """
    if mask is None:
        mask = torch.ones(state.size()[1])

    # normalize by distance between states
    total_distance = torch.sum((state - target_state)**2, 1)

    projected_state = project_to_line(state, target_state, drone_state)

    # divergence from the desired route
    divergence_loss = torch.sum((projected_state - drone_state)**2 * mask, 1)

    # minimize remaining distance to target (normalized on total distance)
    progress_loss_all = (projected_state - target_state)**2 * mask
    progress_loss = torch.sum(progress_loss_all, 1) / total_distance
    if printout:
        print("state", state[0])
        print("target", target_state)
        print("drone", drone_state[0])
        print("progress all", progress_loss_all[0])
        print("divergence", divergence_loss[0].item())
        print("progress_loss", progress_loss[0].item())
        print()
        print(state[1])
        print(target_state)
        print(drone_state[1])
        print(divergence_loss[1].item())
        print(progress_loss[1].item())
        print(fail)
    return .05 * divergence_loss + progress_loss
