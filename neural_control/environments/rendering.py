import numpy as np
from neural_control.environments.helper_simple_env import Euler
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def body_to_world_matrix(euler):
    """
    Creates a transformation matrix for directions from a body frame
    to world frame for a body with attitude given by `euler` Euler angles.
    :param euler: The Euler angles of the body frame.
    :return: The transformation matrix.
    """
    return np.transpose(world_to_body_matrix(euler))


def world_to_body_matrix(euler):
    """
    Creates a transformation matrix for directions from world frame
    to body frame for a body with attitude given by `euler` Euler angles.
    :param euler: The Euler angles of the body frame.
    :return: The transformation matrix.
    """
    roll, pitch, yaw = euler[0], euler[1], euler[2]

    Cy = np.cos(yaw)
    Sy = np.sin(yaw)
    Cp = np.cos(pitch)
    Sp = np.sin(pitch)
    Cr = np.cos(roll)
    Sr = np.sin(roll)

    matrix = np.array(
        [
            [Cy * Cp, Sy * Cp, -Sp],
            [Cy * Sp * Sr - Cr * Sy, Cr * Cy + Sr * Sy * Sp, Cp * Sr],
            [Cy * Sp * Cr + Sr * Sy, Cr * Sy * Sp - Cy * Sr, Cr * Cp]
        ]
    )
    return matrix


def body_to_world(euler, vector):
    """
    Transforms a direction `vector` from body to world coordinates,
    where the body frame is given by the Euler angles `euler.
    :param euler: Euler angles of the body frame.
    :param vector: The direction vector to transform.
    :return: Direction in world frame.
    """
    return np.dot(body_to_world_matrix(euler), vector)


class Renderer:

    def __init__(self, viewer_shape=(500, 500), y_axis=14):
        self.viewer = None
        self.center = None

        self.scroll_speed = 0.1
        self.objects = []
        self.viewer_shape = viewer_shape
        self.y_axis = y_axis

    def draw_line_2d(self, start, end, color=(0, 0, 0)):
        self.viewer.draw_line(start, end, color=color)

    def draw_line_3d(self, start, end, color=(0, 0, 0)):
        self.draw_line_2d((start[0], start[2]), (end[0], end[2]), color=color)

    def draw_circle(
        self, position, radius, color, filled=True
    ):  # pragma: no cover
        from gym.envs.classic_control import rendering
        copter = rendering.make_circle(radius, filled=filled)
        copter.set_color(*color)
        if len(position) == 3:
            position = (position[0], position[2])
        copter.add_attr(rendering.Transform(translation=position))
        self.viewer.add_onetime(copter)

    def draw_polygon(self, v, filled=False):
        from gym.envs.classic_control import rendering
        airplane = rendering.make_polygon(v, filled=filled)
        self.viewer.add_onetime(airplane)

    def add_object(self, new):
        self.objects.append(new)

    def set_center(self, new_center):
        # new_center is None => We are resetting.
        if new_center is None:
            self.center = None
            return

        # self.center is None => First step, jump to target
        if self.center is None:
            self.center = new_center

        # otherwise do soft update.
        self.center = (
            1.0 - self.scroll_speed
        ) * self.center + self.scroll_speed * new_center
        if self.viewer is not None:
            self.viewer.set_bounds(
                -7 + self.center, 7 + self.center, -1, self.y_axis
            )

    def setup(self):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(*self.viewer_shape)

    def render(self, mode='human', close=False):
        if close:
            self.close()
            return

        if self.viewer is None:
            self.setup()

        for draw_ob in self.objects:  # type RenderedObject
            draw_ob.draw(self)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class RenderedObject:

    def draw(self, renderer: Renderer):
        raise NotImplementedError()


class Ground(RenderedObject):  # pragma: no cover

    def __init__(self, step_size=2):
        self._step_size = step_size

    def draw(self, renderer):
        """ Draws the ground indicator.
        """
        center = renderer.center
        renderer.draw_line_2d((-10 + center, 0.0), (10 + center, 0.0))
        pos = round(center / self._step_size) * self._step_size

        for i in range(-8, 10, self._step_size):
            renderer.draw_line_2d((pos + i, 0.0), (pos + i - 2, -2.0))


class QuadCopter(RenderedObject):  # pragma: no cover

    def __init__(self, source):
        self.source = source
        self._show_thrust = True
        self.arm_length = 0.31

    def draw(self, renderer):
        status = self.source._state

        # transformed main axis
        trafo = np.array(
            [status.attitude.roll, status.attitude.pitch, status.attitude.yaw]
        )

        # draw current orientation
        rotated = body_to_world(trafo, [0, 0, self.arm_length / 2])
        renderer.draw_line_3d(status.position, status.position + rotated)

        self.draw_propeller(
            renderer, trafo, status.position, [self.arm_length, 0, 0],
            status.rotor_speeds[0] / 1
        )
        self.draw_propeller(
            renderer, trafo, status.position, [0, self.arm_length, 0],
            status.rotor_speeds[1] / 1
        )
        self.draw_propeller(
            renderer, trafo, status.position, [-self.arm_length, 0, 0],
            status.rotor_speeds[2] / 1
        )
        self.draw_propeller(
            renderer, trafo, status.position, [0, -self.arm_length, 0],
            status.rotor_speeds[3] / 1
        )

    @staticmethod
    def draw_propeller(
        renderer,
        euler,
        position,
        propeller_position,
        rotor_speed,
        arm_length=0.31
    ):
        structure_line = body_to_world(euler, propeller_position)
        renderer.draw_line_3d(position, position + structure_line)
        renderer.draw_circle(
            position + structure_line, 0.2 * arm_length, (0, 0, 0)
        )
        thrust_line = body_to_world(euler, [0, 0, -0.5 * rotor_speed**2])
        renderer.draw_line_3d(
            position + structure_line, position + structure_line + thrust_line
        )


class FixedWingDrone(RenderedObject):

    def __init__(self, source, draw_quad=False):
        self.draw_quad = draw_quad
        self.source = source
        self._show_thrust = True
        self.targets = [[100, 0, 0]]
        self.x_normalize = 0.1
        self.z_offset = 5

    def set_target(self, target):
        self.targets = np.array(target)
        self.x_normalize = 14 / np.max(self.targets[:, 0])

    def draw(self, renderer):
        state = self.source._state.copy()

        # transformed main axis
        trafo = state[6:9] * np.array([1, 1, -1])

        # normalize x to have drone between left and right bound
        # and set z to other way round
        position = [
            -7 + state[0] * self.x_normalize, state[1],
            state[2] * self.x_normalize + self.z_offset
        ]

        # u = state[3]
        # w = state[4]
        # theta = state[4]
        # x_dot = u * np.cos(theta) + w * np.sin(theta)  # forward
        # h_dot = u * np.sin(theta) - w * np.cos(theta)  # upward
        # # theta_vec = np.array([1, 0, np.sin(theta)]) * 3
        # vel_vec = np.array([x_dot, 0, h_dot])
        # vel_vec = vel_vec / np.linalg.norm(vel_vec) * 3
        # renderer.draw_line_3d(position, position + vel_vec, color=(1, 0, 0))

        # draw target point
        for target in self.targets:
            renderer.draw_circle(
                (
                    -7 + target[0] * self.x_normalize,
                    target[2] * self.x_normalize + self.z_offset
                ),
                .2, (0, 1, 0),
                filled=True
            )

        if self.draw_quad:
            self.quad_as_plane(renderer, trafo, position)
        else:
            self.draw_airplane(renderer, position, trafo)

    @staticmethod
    def quad_as_plane(renderer, trafo, position):
        rotated = body_to_world(trafo, [0, 0, 0.5])
        renderer.draw_line_3d(position, position + rotated)

        QuadCopter.draw_propeller(renderer, trafo, position, [1, 0, 0], 0)
        QuadCopter.draw_propeller(renderer, trafo, position, [0, 1, 0], 0)
        QuadCopter.draw_propeller(renderer, trafo, position, [-1, 0, 0], 0)
        QuadCopter.draw_propeller(renderer, trafo, position, [0, -1, 0], 0)

    @staticmethod
    def draw_airplane(renderer, position, euler):
        # plane definition
        offset = np.array([-5, 0, -1.5])
        scale = .3
        coord_plane = (
            np.array(
                [
                    [1, 0, 1], [1, 0, 3.3], [2, 0, 2], [5.5, 0, 2],
                    [5, 0, 2.5], [4.5, 0, 2], [8, 0, 2], [10, 0, 1], [1, 0, 1]
                ]
            ) + offset
        ) * scale

        rot_matrix = body_to_world_matrix(euler)
        coord_plane_rotated = (
            np.array([np.dot(rot_matrix, coord)
                      for coord in coord_plane]) + position
        )[:, [0, 2]]
        renderer.draw_polygon(coord_plane_rotated)

        # add wing
        coord_wing = (
            np.array([[4, 0, 1.5], [5, 0, 0], [6, 0, 1.5]]) + offset
        ) * scale
        coord_wing_rotated = (
            np.array([np.dot(rot_matrix, coord)
                      for coord in coord_wing]) + position
        )[:, [0, 2]]

        renderer.draw_polygon(coord_wing_rotated)


def draw_line_3d(ax, pos1, pos2, color="black"):
    x1, x2, x3 = pos1[0], pos1[1], pos1[2]
    y1, y2, y3 = pos2[0], pos2[1], pos2[2]
    ax.plot3D([x1, y1], [x2, y2], [x3, y3], color=color)
    return ax


def draw_circle(ax, pos, radius, col="green"):
    x, y, z = pos[0], pos[1], pos[2]
    ax.scatter3D(x, y, z, marker="o", s=radius * 100, color=col)
    return ax


def draw_quad(ax, position, trafo):

    # draw current orientation
    rotated = body_to_world(np.array(trafo), np.array([0, 0, 0.5]))
    draw_line_3d(ax, position, position + rotated)

    draw_propeller(ax, trafo, position, [1, 0, 0])
    draw_propeller(ax, trafo, position, [0, 1, 0])
    draw_propeller(ax, trafo, position, [-1, 0, 0])
    draw_propeller(ax, trafo, position, [0, -1, 0])
    return ax


def draw_propeller(ax, euler, position, propeller_position):
    structure_line = body_to_world(euler, propeller_position)
    draw_line_3d(ax, position, position + structure_line)
    draw_circle(ax, position + structure_line, 0.1, (0, 0, 0))
    # thrust_line = body_to_world(euler, [0, 0, -0.5 * rotor_speed**2])
    # draw_line_3d(
    #     ax,
    #     position + structure_line,
    #     position + structure_line + thrust_line,
    #     color="grey"
    # )


def plot_ref(ax, ref):
    # initially: plot full reference
    X_ref = ref[:, 0]
    Y_ref = ref[:, 1]
    Z_ref = ref[:, 2]
    ax.plot3D(X_ref, Y_ref, Z_ref, color="red")
    ax.set_xlim(np.min(X_ref), np.max(X_ref))
    ax.set_ylim(np.min(Y_ref), np.max(Y_ref))
    ax.set_zlim(np.min(Z_ref), np.max(Z_ref))
    return ax


def animate_quad(ref, traj):
    fig = plt.figure(figsize=(5, 5))
    ax = axes3d.Axes3D(fig)

    ax = plot_ref(ax, ref)

    def update(i, ax, fig):
        ax.cla()
        ax = plot_ref(ax, ref)
        ax = draw_quad(ax, traj[i, :3], traj[i, 3:6])
        # s = ax.scatter3D(X1[i], Y1[i], Z1[i], marker="o", c="green", s=100)
        wframe = ax.plot3D(X1[:i], Y1[:i], Z1[:i], color="blue")
        return wframe,

    # trajectory
    X1 = traj[:, 0]
    Y1 = traj[:, 1]
    Z1 = traj[:, 2]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=iter(range(len(ref))),
        fargs=(ax, fig),
        interval=10
    )
    # video = anim.to_html5_video()
    # html = display.HTML(video)
    # display.display(html)
    plt.show()


def draw_fixed_wing(ax, position, euler, stretch=1):
    # line from back to the middle
    back = position + body_to_world(euler, [-stretch, 0, 0])
    draw_line_3d(ax, back, position)
    # line from front to middle-front
    front_middle = position + body_to_world(euler, [stretch * 0.5, 0, 0])
    draw_line_3d(
        ax, position + body_to_world(euler, [stretch * 0.75, 0, 0]), position
    )
    right_back = position + body_to_world(euler, [-stretch * 0.5, 0.5, 0])
    left_back = position + body_to_world(euler, [-stretch * 0.5, -0.5, 0])
    # line from front middle to right back
    draw_line_3d(ax, front_middle, right_back)
    draw_line_3d(ax, front_middle, left_back)
    draw_line_3d(ax, position, right_back)
    draw_line_3d(ax, position, left_back)
    # small line pointing up in the back
    pointing_up = position + body_to_world(euler, [-stretch, 0, 0.25])
    draw_line_3d(ax, back, pointing_up)
    # horizontal line on top
    left_up = position + body_to_world(euler, [-stretch, 0.25, 0.25])
    right_up = position + body_to_world(euler, [-stretch, -0.25, 0.25])
    draw_line_3d(ax, left_up, right_up)
    return ax


def animate_fixed_wing(target_point, traj):
    target_point = np.array(target_point)
    fig = plt.figure(figsize=(10, 8))
    ax = axes3d.Axes3D(fig)
    ax.set_box_aspect(
        (np.ptp(traj[:, 0] / 4), np.ptp(traj[:, 1]), np.ptp(traj[:, 2]))
    )
    # ax.set_box_aspect((np.ptp(10), np.ptp(5), np.ptp(5)))

    # xlim is the maximum point
    ax.set_xlim(-1, target_point[-1, 0])
    ax.set_ylim(-7, 7)
    ax.set_zlim(-7, 7)
    s = ax.scatter3D(
        target_point[:, 0],
        target_point[:, 1],
        target_point[:, 2],
        marker="o",
        c="green",
        s=100
    )

    def update(i, ax, fig):
        ax.cla()
        euler = traj[i, 6:9] * np.array([1, 1, -1])
        s = ax.scatter3D(
            target_point[:, 0],
            target_point[:, 1],
            target_point[:, 2],
            marker="o",
            c="green",
            s=100
        )

        ax = draw_fixed_wing(ax, traj[i, :3], euler)
        ax.set_xlim(-1, target_point[-1, 0])
        ax.set_ylim(-7, 7)
        ax.set_zlim(-7, 7)

    ax.view_init(elev=10., azim=270)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=iter(range(len(traj))),
        fargs=(ax, fig),
        interval=10
    )
    plt.show()
