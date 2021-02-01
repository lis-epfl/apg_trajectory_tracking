import numpy as np
from .plan_trajectory import get_reference
import sys
sys.path.append("..")
from environments.rendering import StraightObject


class Hover:

    def __init__(self, drone_state, *args, **kwargs):
        self.target_pos = drone_state[:3]
        self.dt = kwargs["dt"]
        self.horizon = kwargs["horizon"]

    def get_ref_traj(self, drone_state, drone_acc):
        pos = drone_state[:3]
        vel = drone_state[6:9]
        trajectory = get_reference(
            pos,
            vel,
            drone_acc,
            self.target_pos,
            np.zeros(3),
            delta_t=self.dt,
            ref_length=self.horizon
        )
        return trajectory

    def project_on_ref(self, drone_state):
        return self.target_pos


class Straight:

    def __init__(
        self,
        drone_state,
        render,
        renderer,
        max_drone_dist=0.25,
        horizon=10,
        dt=0.05,
        **kwargs
    ):
        """
        Make trajectory from velocity
        """
        traj_direction = drone_state[6:9].copy()
        a_on_line = drone_state[:3].copy()
        b_on_line = a_on_line + traj_direction / np.linalg.norm(traj_direction)

        if render:
            renderer.add_object(
                StraightObject(a_on_line, 5 * b_on_line - 4 * a_on_line)
            )
        self.a_on_line = a_on_line
        self.b_on_line = b_on_line
        self.dt = dt
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist

    def get_ref_traj(self, drone_state, drone_acc):
        """
        Given a straight reference between A and B, compute the next x ref states
        b_on_line - a_on_line must be a unit vector!
        """
        drone_pos = drone_state[:3]
        projected = np_project_line(self.a_on_line, self.b_on_line, drone_pos)
        direction = self.b_on_line - self.a_on_line
        # norm squared is a^2
        dist1 = np.sum((projected - drone_pos)**2)
        # a^2 + b^2 = max_drone_dist^2
        dist_on_line = np.sqrt(max([self.max_drone_dist**2 - dist1, 0]))
        goal_pos = projected + direction * dist_on_line
        reference = get_reference(
            drone_pos,
            drone_state[6:9],
            drone_acc,
            goal_pos,
            direction,
            ref_length=self.horizon,
            delta_t=self.dt
        )
        return reference

    def project_on_ref(self, drone_state):
        """
        Project the current state to the trajectory
        """
        return np_project_line(self.a_on_line, self.b_on_line, drone_state[:3])


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
