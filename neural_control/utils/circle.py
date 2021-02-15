import numpy as np
from .plan_trajectory import get_reference


class Circle:
    """
    Auxiliary class to sample from a circular reference trajectory
    """

    def __init__(
        self,
        drone_state,
        render=False,
        renderer=None,
        radius=1,
        plane=[0, 1],
        direction=1,
        max_drone_dist=.25,
        horizon=10,
        dt=0.02
    ):
        """
        initialize a circle with a center and radius
        Arguments:
            mid_point: list of len 3 or None (initialised later from drone pose)
            radius: float>0
            plane: circle lies in a 2D plane of a 3D coordinate system. plane
                argument is tuple of the two axes, while the third one is fixed
        """
        self.plane = plane
        self.radius = radius
        self.direction = direction
        self.horizon = horizon
        self.max_drone_dist = max_drone_dist
        self.dt = dt
        # init renderer
        self.init_from_tangent(drone_state[:3], drone_state[6:9])
        if render:
            if renderer is None:
                raise ValueError("if render is true, need to input renderer")
            renderer.add_object(CircleObject(self.mid_point, radius))

    def init_from_tangent(self, pos, vel):
        """
        Initialize the center by the current drone position
        Arguments:
            pos: np array of length 3
            vel: np array of length 3
        """
        # fixed axis will just stay
        mid_point_tmp = pos.astype(float)
        # get 2D vel
        vel_2D = vel[self.plane]
        if np.all(np.isclose(vel_2D, 0)):
            vel_2D = np.random.rand(2) - .5
        # get orthogonal vector pointing to middle of circle
        orthogonal_vec = np.array([(-1) * vel_2D[1], vel_2D[0]])
        # compute center
        unit_vec = orthogonal_vec / np.linalg.norm(orthogonal_vec)
        mid_point_2D = pos[self.plane
                           ] + unit_vec * self.radius * self.direction
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

    def project_point(self, point, addon=0):
        point_2D = self.to_2D(point)
        alpha = self.to_alpha(point_2D) + addon * self.direction
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
        alpha = (self.to_alpha(point_2D) +
                 alpha_between * self.direction) % (2 * np.pi)
        target_point = self.point_on_circle(alpha)
        return self.to_3D(target_point)

    def get_velocity(self, point_3D, stepsize=0.1):
        """
        Compute the tangent to a point
        """
        point_2D = self.to_2D(point_3D)
        curr_alpha = self.to_alpha(point_2D)
        next_alpha = curr_alpha + stepsize * self.direction
        next_point = self.point_on_circle(next_alpha)
        return self.to_3D(next_point) - point_3D

    def project_on_ref(self, point):
        return self.to_3D(self.project_point(point, addon=0))

    def get_ref_traj(self, drone_state, drone_acc):
        drone_pos = drone_state[:3]
        goal_pos = self.next_target(drone_pos, self.max_drone_dist)
        direction = self.get_velocity(goal_pos)
        reference = get_reference(
            drone_pos,
            drone_state[6:9],
            drone_acc,
            goal_pos,
            direction,
            ref_length=self.horizon,
            delta_t=self.dt
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


class CircleObject():

    def __init__(self, mid_point, radius):
        self.mid_point = mid_point.copy()
        self.mid_point[2] += 1
        self.radius = radius

    def draw(self, renderer):
        renderer.draw_circle(
            tuple(self.mid_point), self.radius, (0, 1, 0), filled=False
        )