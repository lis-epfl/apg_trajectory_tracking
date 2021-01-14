import numpy as np

# TODO: compute attitude velocities etc

def positions_to_state_trajectory(drone_state, ref_positions):
    """
    Compute full reference trajectory from given drone state and
    target positions
    Arguments:
        drone_state: vector of size s (state dimension),
            full state of the drone
        ref_positions: array of size (x,3) with the x next target
            positions in 3D space.
    Returns:
        Array of size (x, s) with the x next reference states
    """
    pass

def straight_traj(drone_pos):
    # TODO: execute one step and make this to a straight trajectory
    end = np.random.rand(3)-0.5

def sample_points_on_straight(current_ref_point, direction, step_size=0.2, ref_length=5):
    reference_states = np.zeros((ref_length, 3))
    dist = np.sqrt(np.sum(direction**2))
    # one step is step_size times unit vector
    step_dir = direction / dist * step_size
    for i in range(ref_length):
        reference_states[i] = current_ref_point + (i)* step_dir
    return reference_states

def np_project_line(a, b, p):
    """
    Project point p on line spanned by a and b
    """
    if np.all(a==b):
        return a
    ap = p-a
    ab = np.expand_dims(b-a, 1)
    dot = np.dot(ab, ab.T)
    norm = np.sum(ab**2)
    result = a + np.dot(dot, ap) / norm
    return result

def straight_training_sample(training=False, step_size=0.2, max_drone_dist=0.1, ref_length=5):
    """
    Sample necessary training data for training a straight trajectory
    """
    # sample trajectory
    start = np.random.rand(3)-0.5
    end = np.random.rand(3)-0.5
    connection = end - start
    point_on_traj = start + np.random.rand() * connection
    # drone is in position in slight divergence from the trajectory
    pos_drone = point_on_traj + (np.random.rand(3) *2 -1) * max_drone_dist
    reference_states = sample_points_on_straight(
        point_on_traj, connection, step_size=step_size, ref_length=ref_length
    )
    # have reference in relation to drone state
    reference_states = reference_states - pos_drone
    return reference_states

def sample_to_input(drone_state, reference_states):
    """
    Takes absolute states and returns them as suitable input to a neural network
    Arguments:
        drone_state: np array of size s_dim, full state of drone (unnormalized!)
        reference_state: np array of size (x, 3) containing target positions
    """
    reference_states = (reference_states - drone_state[:3]).flatten()
    return reference_states