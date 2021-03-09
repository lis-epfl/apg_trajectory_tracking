import torch
from neural_control.environments.drone_dynamics import simulate_quadrotor, device
torch.autograd.set_detect_anomaly(True)
zero_tensor = torch.zeros(3).to(device)


def drone_loss_function(current_state, start_state=None, printout=0):
    """
    Computes loss of the current state (target is assumed to be zero-state)
    Arguments:
        current_state: array with x entries describing attitude and velocity
        start_state: same format as current_state, but start position (for
            normalization)
    Returns:
        loss (torch float scalar)
    """
    # weighting
    angle_factor = 1
    angvel_factor = 2e-2
    pos_factor = .5

    # attittude and att velocity loss
    angle_error = torch.sum(current_state[:, 3:6]**2, axis=1)
    ang_vel_error = torch.sum(current_state[:, 12:]**2, axis=1)

    # position loss
    div, prog = pos_traj_loss(start_state[:, :3], current_state[:, :3])
    position_loss = div + 3 * prog  # added 3 only
    # torch.sum(
    #     current_state[:, :3]**2 * torch.tensor([.5, 2, 2]), dim=1
    # )

    # angle_factor = torch.relu(angle_factor - position_loss)

    # together
    loss = (
        angle_factor * angle_error + angvel_factor * ang_vel_error +
        pos_factor * position_loss
    )

    if printout:
        print()
        print("attitude loss", (angle_factor * angle_error)[0])
        print("att vel loss", (angvel_factor * ang_vel_error)[0])
        print("position loss", (pos_factor * position_loss)[0])
    return torch.sum(loss)


def reference_loss(states, ref_states, printout=0, delta_t=0.02):
    """
    Compute loss with respect to reference trajectory
    """
    # TODO: add loss on actions with quaternion formulation
    # (9.81, 0,0,0)
    # TODO: include attitude in reference
    angle_factor = 0.01
    angvel_factor = 2e-2
    vel_factor = 0.05
    pos_factor = 10
    yaw_factor = 10

    position_loss = torch.sum((states[:, :, :3] - ref_states[:, :, :3])**2)
    velocity_loss = torch.sum((states[:, :, 6:9] - ref_states[:, :, 3:6])**2)

    angle_error = 0
    for k in range(states.size()[1] - 2):
        # approximate acceleration
        acc = (states[:, k + 1, 6:9] - states[:, k, 6:9]) / delta_t
        acc_ref = ref_states[:, k, 6:9]
        # subtract from desired acceleration
        angle_error += torch.sum((acc_ref - acc)**2)

    ang_vel_error = torch.sum(states[:, :, 9:11]**2
                              ) + yaw_factor * torch.sum(states[:, :, 11]**2)

    loss = (
        angle_factor * angle_error + angvel_factor * ang_vel_error +
        pos_factor * position_loss + vel_factor * velocity_loss
    )

    if printout:
        print()
        print("attitude loss", (angle_factor * angle_error).item())
        print("att vel loss", (angvel_factor * ang_vel_error).item())
        print("velocity loss", (velocity_loss * vel_factor).item())
        print("position loss", (pos_factor * position_loss).item())
    return loss


def project_to_line(a_on_line, b_on_line, p):
    """
    Project a point p to a line from a to b
    Arguments:
        All inputs are 2D tensors of shape (BATCH_SIZE, n)
        a_on_line: First point on the line
        b_on_line: Second point on the line
        p: point to be projected onto the line
    Returns: Tensor of shape (BATCH_SIZE, n) which is the orthogonal projection
            of p on the line
    """
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


def pos_traj_loss(start_state, drone_state):
    """
    Compute position loss based on the projection of the drone state on the
    target trajectory (from start_state to zero)
    Arguments: (Shape BATCH_SIZE, 3)
        start_state: Drone position before action
        drone_state: Position after applying action to start_state
    Returns:
        Two tensors, each of shape (BATCH_SIZE, 1)
        divergence_loss: divergence from the target trajectory
        progress_loss: 1 - how far did the drone progress towards the target
    """
    # distance from start to target
    total_distance = torch.sum(start_state**2, 1)
    # project to trajectory
    projected_state = project_to_line(start_state, zero_tensor, drone_state)
    # losses
    divergence_loss = torch.sum(
        (projected_state - drone_state)**2, 1
    ) / total_distance
    progress_loss = torch.sum(projected_state**2, 1) / total_distance
    return divergence_loss, progress_loss


action_prior = torch.tensor([.25, .5])


def fixed_wing_loss(drone_states, linear_reference, actions, printout=0):
    div_weight = torch.tensor([1, 100])
    pitch_weight = 0
    action_weight = 1

    divergence_loss = torch.sum(
        (drone_states[:, :, :2] - linear_reference)**2 * div_weight
    ) * 0.1

    # pitch loss:
    pitch_loss = torch.sum(drone_states[:, :, 5]**2)

    # action loss
    action_loss = torch.sum((actions - action_prior)**2)

    # together
    loss = divergence_loss + (pitch_weight *
                              pitch_loss) + (action_weight * action_loss)
    if printout:
        import numpy as np
        print(linear_reference.size(), drone_states.size())
        np.set_printoptions(precision=3, suppress=True)
        print("target")
        print(linear_reference.detach().numpy()[0])
        print("drone states")
        print(drone_states.detach().numpy()[0])
        exit()
    return loss


def trajectory_loss(
    start_state, target_state, drone_state, actions, printout=0
):
    """
    Trajectory loss for position and attitude (in contrast to pos_traj_loss)
    Input states must be normalized!
    """
    div_weight = torch.tensor([1, 1000])
    action_weight = 10
    pitch_weight = 0

    start_pos = start_state[:, :2]
    target_pos = target_state[:, :2]
    drone_pos = drone_state[:, :2]

    # divergence_loss = 0
    # for k in range(drone_state.size()[1]):
    #     drone_pos = drone_state[:, k, :2]

    # normalize by distance between states
    # total_distance = (start_pos - target_pos)**2
    projected_pos = project_to_line(start_pos, target_pos, drone_pos)
    # projected_states[:, k] = projected_pos

    # divergence from the desired route
    divergence_loss_all = (projected_pos - drone_pos)**2 * div_weight
    divergence_loss = torch.sum(divergence_loss_all)

    # pitch loss:
    pitch_loss = torch.sum(drone_state[:, 5]**2)

    # action loss
    action_loss = torch.sum((actions - action_prior)**2)

    # together
    loss = divergence_loss + (pitch_weight *
                              pitch_loss) + (action_weight * action_loss)

    # minimize remaining distance to target (normalized on total distance)
    # progress_loss_all = (projected_pos - target_pos)**2
    # progress_loss = torch.sum(progress_loss_all / total_distance)

    # # penalize angular velocity
    # av_loss = torch.sum(drone_state[:, 5]**2)

    if printout:
        import numpy as np
        np.set_printoptions(precision=3, suppress=True)
        print("start")
        print(start_state.detach().numpy()[0])
        print("drone states")
        print(drone_state.detach().numpy()[0])
        print("target pos")
        print(target_state.detach().numpy()[0])
        print("projected pos")
        print(projected_pos.detach().numpy()[0])
        exit()
    return loss
    # (
    #     10 * (
    #         pro_weight * progress_loss + div_weight * divergence_loss +
    #         av_weight * av_loss
    #     )
    # )
