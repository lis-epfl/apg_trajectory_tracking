import numpy as np
from .plan_trajectory import RapidTrajectory


def get_reference(pos0, vel0, acc0, posf, velf, delta_t=0.02, ref_length=5):
    """
    Compute reference trajectory based on start (0) and final (f) states
    """
    traj = RapidTrajectory(pos0, vel0, acc0, [0, 0, -9.81])
    traj.set_goal_position(posf)
    traj.set_goal_velocity(velf)
    traj.set_goal_acceleration([0, 0, 0])
    # Run the algorithm, and generate the trajectory.
    traj.generate(delta_t * ref_length)
    # # Test input feasibility
    # fmin = 5  #[m/s**2]
    # fmax = 25 #[m/s**2]
    # wmax = 20 #[rad/s]
    # min_time = 0.02 #[s]
    # inputsFeasible = traj.check_input_feasibility(fmin, fmax, wmax, min_time)

    # output reference of pos, vel, and acc
    ref_states = np.zeros((ref_length, 9))
    for j, timepoint in enumerate(np.arange(0, ref_length * delta_t, delta_t)):
        # print(t, traj.get_velocity(t))
        ref_states[j] = np.concatenate(
            (
                traj.get_position(timepoint), traj.get_velocity(timepoint),
                traj.get_acceleration(timepoint)
            )
        )
    return ref_states


def eval_get_reference(
    drone_state, drone_acc, a_on_line, b_on_line, max_drone_dist, ref_length
):
    """
    Given a straight reference between A and B, compute the next x ref states
    b_on_line - a_on_line must be a unit vector!
    """
    drone_pos = drone_state[:3]
    projected = np_project_line(a_on_line, b_on_line, drone_pos)
    direction = b_on_line - a_on_line
    # norm squared is a^2
    dist1 = np.sum((projected - drone_pos)**2)
    # a^2 + b^2 = max_drone_dist^2
    dist_on_line = np.sqrt(max([max_drone_dist**2 - dist1, 0]))
    goal_pos = projected + direction * dist_on_line
    reference = get_reference(
        drone_pos,
        drone_state[6:9],
        drone_acc,
        goal_pos,
        direction,
        ref_length=ref_length
    )
    # TODO: direction times some factor?
    return reference


# TODO: compute attitude velocities etc


def positions_to_state_trajectory(drone_state, ref_positions, delta_t=0.02):
    """
    Compute full reference trajectory from given drone state and
    target positions -> for testing
    Arguments:
        drone_state: vector of size s (state dimension),
            full state of the drone
        ref_positions: array of size (x,3) with the x next target
            positions in 3D space.
    Returns:
        Array of size (x, s) with the x next reference states
    """
    # get appropriate ref state as goal
    # project drone to ref traj
    # get reference
    reference = get_reference(
        drone_state[:3],
        drone_state[6:9],
    )


def sample_points_on_straight(
    current_ref_point, direction, step_size=0.2, ref_length=5
):
    """
    Get the next x reference points in direction direction starting from
    current_ref_point
    Arguments:
        current_ref_point: coordinates of the start position of the reference
        direction: 3D vector with direction of reference
        step_size: distance between subsequent states
        ref_length: Number of states to sample (horizon)
    """
    reference_states = np.zeros((ref_length, 3))
    dist = np.sqrt(np.sum(direction**2))
    # one step is step_size times unit vector
    step_dir = direction / dist * step_size
    for i in range(ref_length):
        reference_states[i] = current_ref_point + (i) * step_dir
    return reference_states


def np_project_line(a, b, p):
    """
    Project point p on line spanned by a and b
    """
    if np.all(a == b):
        return a
    ap = p - a
    ab = np.expand_dims(b - a, 1)
    dot = np.dot(ab, ab.T)
    norm = np.sum(ab**2)
    result = a + np.dot(dot, ap) / norm
    return result


def straight_training_sample(step_size=0.2, max_drone_dist=0.1, ref_length=5):
    """
    Sample necessary training data for training a straight trajectory
    """
    # sample trajectory
    start = np.random.rand(3) - 0.5
    end = np.random.rand(3) - 0.5
    connection = end - start
    point_on_traj = start + np.random.rand() * connection
    # drone is in position in slight divergence from the trajectory
    pos_drone = point_on_traj + (np.random.rand(3) * 2 - 1) * max_drone_dist
    reference_states = sample_points_on_straight(
        point_on_traj, connection, step_size=step_size, ref_length=ref_length
    )
    # have reference in relation to drone state
    reference_states = reference_states - pos_drone
    return reference_states


def sample_to_input(drone_state, reference_states):
    """
    Takes absolute states and returns them as suitable input to a neural net
    Arguments:
        drone_state: np array of size s_dim, full state of drone (unnormalized)
        reference_state: np array of size (x, 3) containing target positions
    """
    reference_states = (reference_states - drone_state[:3])
    return reference_states
