import numpy as np
from .trajectory import get_reference


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
        alpha = self.to_alpha(point_2D) + 0.2
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
