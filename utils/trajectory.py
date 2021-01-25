import numpy as np
from .plan_trajectory import RapidTrajectory


class Circle:
    """
    Auxiliary class to sample from a circular reference trajectory
    """

    def __init__(self, mid_point=None, radius=1, plane=[0, 1]):
        """
        axis: fixed axis, circle is in plane of this point
        """
        self.plane = plane
        self.radius = radius
        self.mid_point = np.array(mid_point)

    def init_from_tangent(self, pos, vel):
        # fixed axis will just stay
        mid_point_tmp = pos.astype(float)
        # get 2D vel
        vel_2D = vel[self.plane]
        # get orthogonal vector pointing to middle of circle
        orthogonal_vec = np.array([(-1) * vel_2D[1], vel_2D[0]])
        # compute center
        unit_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)
        mid_point_2D = pos[self.plane] + unit_vec * self.radius
        mid_point_tmp[self.plane] = mid_point_2D
        self.mid_point = mid_point_tmp

    def to_2D(self, point):
        return (point - self.mid_point)[self.plane]

    def to_3D(self, point):
        point_3D = np.zeros(3)
        point_3D[self.plane] = point
        return point_3D + self.mid_point

    def to_alpha(self, point_2D):
        x, y = tuple(point_2D)
        if x == 0:
            alpha = np.pi * 0.5
        else:
            alpha = np.arctan(y / x)
        if x < 0:
            alpha += np.pi
        elif y < 0:
            alpha += 2 * np.pi
        return alpha

    def point_on_circle(self, alpha):
        return np.array(
            [np.cos(alpha) * self.radius,
             np.sin(alpha) * self.radius]
        )

    def project_point(self, point):
        point_2D = self.to_2D(point)
        alpha = self.to_alpha(point_2D)
        projected = self.point_on_circle(alpha)
        return projected

    def next_target(self, point, dist_3D):
        """
        Get next target on circle with given maximum distance
        """
        projected = self.to_3D(self.project_point(point))
        dist_to_circle = np.linalg.norm(point - projected)
        if dist_to_circle >= dist_3D:
            # simply return projection to circle as target
            return projected

        # subtract the distance to the plane
        mask = np.ones(3)
        mask[self.plane] = 0
        dist_to_plane = np.sum((point - self.mid_point) * mask)
        dist = np.sqrt(dist_3D**2 - dist_to_plane**2)
        # project to 2D circle
        point_2D = self.to_2D(point)
        dist_from_center = np.linalg.norm(point_2D)
        # cosine rule
        cos_alpha = (self.radius**2 + dist_from_center**2 -
                     dist**2) / (2 * dist_from_center * self.radius)
        alpha_between = np.arccos(cos_alpha)
        alpha = (self.to_alpha(point_2D) + alpha_between) % (2 * np.pi)
        target_point = self.point_on_circle(alpha)
        return self.to_3D(target_point)

    def get_velocity(self, point_3D, stepsize=0.1):
        """
        Compute the tangent to a point
        """
        point_2D = self.to_2D(point_3D)
        curr_alpha = self.to_alpha(point_2D)
        next_alpha = curr_alpha + stepsize
        next_point = self.point_on_circle(next_alpha)
        return self.to_3D(next_point)

    def project_helper(self, point):
        return self.to_3D(self.project_point(point))

    def eval_get_circle(
        self, drone_state, drone_acc, max_drone_dist, ref_length
    ):
        drone_pos = drone_state[:3]
        goal_pos = self.next_target(drone_pos, max_drone_dist)
        direction = self.get_velocity(goal_pos)
        reference = get_reference(
            drone_pos,
            drone_state[6:9],
            drone_acc,
            goal_pos,
            direction,
            ref_length=ref_length
        )
        # TODO: distance some factor?
        return reference

    def plot_circle(self):
        plt.figure(figsize=(5, 5))
        points = npa(
            [
                self.mid_point[self.plane] + self.point_on_circle(alpha)
                for alpha in np.arange(0, 2 * np.pi, 0.2)
            ]
        )
        plt.scatter(points[:, 0], points[:, 1])


def get_reference(pos0, vel0, acc0, posf, velf, delta_t=0.02, ref_length=5):
    """
    Compute reference trajectory based on start (0) and final (f) states
    """
    # generate trajectory
    traj = RapidTrajectory(pos0, vel0, acc0, [0, 0, -9.81])
    traj.set_goal_position(posf)
    traj.set_goal_velocity(velf)
    traj.set_goal_acceleration([0, 0, 0])
    # Run the algorithm, and generate the trajectory.
    traj.generate(delta_t * ref_length)
    # add 1 because otherwise the current state of the drone counts as well
    ref_length += 1
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
    # exclude the current state
    return ref_states[1:]


def eval_get_straight_ref(
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
