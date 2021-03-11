import numpy as np
from scipy.stats import special_ortho_group
from .plan_trajectory import get_reference
from scipy.interpolate import CubicSpline


class Polynomial:

    def __init__(
        self,
        drone_state,
        render=False,
        renderer=None,
        points_to_traverse=None,
        max_drone_dist=0.25,
        horizon=10,
        hover_steps=50,
        x_range=10,
        degree=8,
        dt=0.05,
        **kwargs
    ):
        """
        Create random trajectory
        """
        dist_points = max_drone_dist / horizon
        self.hover_steps = hover_steps
        self.dist_points = dist_points
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.dt = dt
        # make variable whether we are already finished with the trajectory
        self.finished = False
        if render and renderer is None:
            raise ValueError("if render is true, need to input renderer")

        if points_to_traverse is None:
            points_3d = self.random_polynomial(x_range, degree)
        else:
            points_3d = self.cubic_fit(points_to_traverse)

        # subtract current position to start there
        points_3d = points_3d - points_3d[0] + drone_state[:3]

        start_hover = np.array([points_3d[0] for _ in range(hover_steps)])
        end_hover = np.array([points_3d[-1] for _ in range(hover_steps)])

        self.reference = np.vstack([start_hover, points_3d, end_hover])
        self.ref_len = len(self.reference)
        self.target_ind = 0
        self.current_ind = 0

        # draw trajectory on renderer
        if render:
            renderer.add_object(PolyObject(self.reference))

    def cubic_fit(self, points_to_traverse):
        dists = [0] + [
            np.linalg.norm(points_to_traverse[i] - points_to_traverse[i + 1])
            for i in range(len(points_to_traverse) - 1)
        ]
        cum_dists = np.cumsum(dists)

        # add one dummy point to prevent fast speed in the beginning
        add_point_bef = points_to_traverse[1]
        rand_vec_2 = np.random.rand(3) * 2 - 1
        add_point_aft = points_to_traverse[-1] - rand_vec_2
        x = np.array(
            [-1 * dists[1]] + cum_dists.tolist() +
            [cum_dists[-1] + np.linalg.norm(add_point_aft)]
        )
        fit_points = np.vstack(
            (add_point_bef, points_to_traverse, add_point_aft)
        )

        # fit cubic spline
        func = CubicSpline(x, fit_points)

        # sample in steps dependent on max_drone_dist
        x_sample = np.arange(0, cum_dists[-1], self.dist_points)
        points_sample = np.array([func(x_s) for x_s in x_sample])
        return points_sample

    def random_polynomial(self, x_range, degree):
        x_start = 1
        x_final = x_start + x_range

        # generate random data
        x = np.linspace(x_start - 1, x_final + 1, 10)
        y = np.random.rand(len(x)) * 5 + 5

        # generate random rotation to 3D
        rot = special_ortho_group.rvs(3)

        # fit polynomial
        poly = np.poly1d(np.polyfit(x, y, degree))

        # define gradient function
        get_gradient = lambda x: np.sum(
            [
                (degree - i) * poly.coefficients[i] * x**(degree - i - 1)
                for i in range(degree)
            ]
        )

        points_2d = [[x_start, poly(x_start)]]
        while x_start < x_final:
            # start at x = 0
            grad = get_gradient(x_start)
            vec = np.array([1, grad])
            normed_vec = vec / np.linalg.norm(vec)
            x_end = x_start + (normed_vec * self.dist_points)[0]
            points_2d.append([x_end, poly(x_end)])
            x_start = x_end
            # print([x_end, poly(x_end)])
        points_2d = np.array(points_2d)
        points_2d_ext = np.swapaxes(
            np.vstack(
                [points_2d[:, 0],
                 np.zeros(len(points_2d)), points_2d[:, 1]]
            ), 1, 0
        )

        # transform to 3D
        points_3d = points_2d_ext @ rot
        return points_3d

    def get_fixed_ref(self, drone_state, drone_acc):
        """
        Return directly the points on the reference, and not the relative min
        snap
        Working with MPC, but not with neural controller
        """
        out_reference = np.zeros((self.horizon, 9))
        # if already at end, return zero velocities and accelerations
        if self.current_ind >= len(self.reference) - self.horizon - 2:
            out_reference[:, :3] = self.reference[self.current_ind]
            return out_reference
        # else: compute next velocities and accs
        next_positions = self.reference[self.current_ind:self.current_ind +
                                        self.horizon + 2]
        next_velocities = [
            (next_positions[i + 1] - next_positions[i]) / self.dt * 2
            for i in range(self.horizon + 1)
        ]
        next_accs = [
            (next_velocities[i + 1] - next_velocities[i]) / self.dt
            for i in range(self.horizon)
        ]
        ref_out = np.hstack(
            (
                next_positions[:self.horizon], next_velocities[:self.horizon],
                next_accs[:self.horizon]
            )
        )
        self.current_ind += 1
        self.target_ind += 1
        # np.set_printoptions(suppress=True, precision=3)
        # print(ref_out)
        return ref_out

    def get_ref_traj(self, drone_state, drone_acc):
        """
        Given the current position, compute a min snap trajectory to the next
        target
        """
        drone_pos = drone_state[:3]
        drone_vel = drone_state[6:9]

        # if we are close enough to the next point, set it as the next target,
        # otherwise stick with the current target
        if self.target_ind < self.ref_len - 2:
            dist_from_next = np.linalg.norm(
                drone_pos - self.reference[self.target_ind + 1]
            )
            if dist_from_next < self.max_drone_dist:
                self.target_ind += 1
        else:
            self.finished = True
        goal_pos = self.reference[self.target_ind]

        # TODO: is the velocity simply the two subtracted? or times dt or so?
        goal_vel = self.reference[self.target_ind +
                                  1] - self.reference[self.target_ind]

        reference = get_reference(
            drone_pos,
            drone_vel,
            drone_acc,
            goal_pos,
            goal_vel,
            ref_length=self.horizon,
            delta_t=self.dt
        )
        return reference

    def project_on_ref(self, drone_state):
        """
        Project drone state onto the trajectory
        """
        start = max([self.target_ind - 20, 0])
        end = max([self.target_ind, 2])
        possible_locs = self.reference[start:end]
        # compute distance to each of them
        distances = [
            np.linalg.norm(drone_state[:3] - loc) for loc in possible_locs
        ]
        # return the closest one
        return possible_locs[np.argmin(distances)]


class PolyObject():

    def __init__(self, reference_arr):
        self.points = np.array(
            [
                reference_arr[i] for i in range(len(reference_arr))
                if i % 20 == 0
            ]
        )
        self.points[:, 2] += 1

    def draw(self, renderer):
        for p in range(len(self.points) - 1):
            renderer.draw_line_3d(
                self.points[p], self.points[p + 1], color=(1, 0, 0)
            )
