import torch
from environments.drone_dynamics import simulate_quadrotor
torch.autograd.set_detect_anomaly(True)


def drone_loss_function(current_state, printout=0, pos_factor=1, start_dist=1):
    """
    Computes loss for applying an action to the current state by comparing to
    the target state
    Arguments:
        current_state: array with x entries describing attitude and velocity
        action: control signal of dimension 4 (thrust of rotors)
    """
    # weighting
    angle_factor = 10.0
    angvel_factor = 1
    pos_factor = .05

    # attittude and att velocity loss
    angle_error = torch.sum(current_state[:, 3:6]**4, axis=1)
    ang_vel_error = torch.sum(current_state[:, 13:16]**2, axis=1)

    # position loss
    position_loss = torch.sum(current_state[:, :3]**2, dim=1) / start_dist

    # angle_factor = torch.relu(angle_factor - position_loss)

    # together
    loss = (
        angle_factor * angle_error + angvel_factor * ang_vel_error +
        pos_factor * position_loss
    )

    if printout:
        print()
        print("factor", angle_factor)
        print("attitude loss", (angle_factor * angle_error)[0])
        print("att vel loss", (angvel_factor * ang_vel_error)[0])
        print("position loss", (pos_factor * position_loss)[0])
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


def trajectory_loss(
    state,
    target_state,
    drone_state,
    loss_weights=None,
    mask=None,
    printout=0
):
    """
    Loss for attemtping to traverse from state to target_state but ending up
    at drone_state
    """
    if mask is None:
        mask = torch.ones(state.size()[1])
    else:
        state = state * mask

    # multiply losses by 0 or weights
    mask = mask * loss_weights

    # normalize by distance between states
    total_distance = torch.sum((state - target_state)**2, 1)

    projected_state = project_to_line(state, target_state, drone_state)

    # divergence from the desired route
    divergence_loss_all = (projected_state - drone_state)**2 * mask
    divergence_loss = torch.sum(divergence_loss_all, 1)

    # minimize remaining distance to target (normalized on total distance)
    progress_loss_all = (projected_state - target_state)**2 * mask
    progress_loss = torch.sum(progress_loss_all, 1) / total_distance
    if printout:
        print(
            total_distance.size(), progress_loss_all.size(),
            progress_loss.size()
        )
        print("state", state[0])
        print("target", target_state)
        print("drone", drone_state[0])
        print("divergence all", divergence_loss_all[0])
        print("progress all", progress_loss_all[0])
        print("divergence", divergence_loss[0].item())
        print("progress_loss", progress_loss[0].item())
        print("final", progress_loss + .1 * divergence_loss)
        print(fail)
    return torch.sum(progress_loss + .1 * divergence_loss)
