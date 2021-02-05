import numpy as np
from scipy.stats import special_ortho_group
from .trajectory import get_reference


class RandomReference:

    def __init__(
        self,
        drone_state,
        render,
        renderer,
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
        self.hover_steps = 50
        self.dist_points = dist_points
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.dt = dt

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
            x_end = x_start + (normed_vec * dist_points)[0]
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

        # for visibility, bring x and z in a good range
        # x_min_render, z_min_render = (-3, 0.5)
        # x_max_render, z_max_render = (3, 7)
        points_3d = points_3d - points_3d[0] + drone_state[:3]
        # points_3d[:,0] + x_min_render - np.min(points_3d[:, 0])
        # points_3d[:,2] = points_3d[:,
        #                          2] + z_min_render - np.min(points_3d[:, 2])

        start_hover = np.array([points_3d[0] for _ in range(hover_steps)])
        end_hover = np.array([points_3d[-1] for _ in range(hover_steps)])

        self.reference = np.vstack([start_hover, points_3d, end_hover])
        self.ref_len = len(self.reference)
        self.target_ind = 0

        # draw trajectory on renderer
        if render:
            renderer.add_object(PolyObject(self.reference))

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
