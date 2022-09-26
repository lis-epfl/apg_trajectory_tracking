import numpy as np
from neural_control.environments.cartpole_rendering import LineStyle
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


def draw_quad(ax, position, trafo, c="black"):

    # draw current orientation
    arm_length = 0.31
    rotated = body_to_world(np.array(trafo), np.array([0, 0, arm_length / 2]))
    draw_line_3d(ax, position, position + rotated, color=c)

    draw_propeller(ax, trafo, position, [arm_length, 0, 0], c=c)
    draw_propeller(ax, trafo, position, [0, arm_length, 0], c=c)
    draw_propeller(ax, trafo, position, [-arm_length, 0, 0], c=c)
    draw_propeller(ax, trafo, position, [0, -arm_length, 0], c=c)
    return ax


def draw_propeller(ax, euler, position, propeller_position, c="black"):
    arm_length = 0.31
    structure_line = body_to_world(euler, propeller_position)
    draw_line_3d(ax, position, position + structure_line, color=c)
    draw_circle(ax, position + structure_line, 0.05 * arm_length, col=c)
    # # For plotting thrust:
    # thrust_line = body_to_world(euler, [0, 0, -0.5 * rotor_speed**2])
    # draw_line_3d(
    #     ax,
    #     position + structure_line,
    #     position + structure_line + thrust_line,
    #     color="grey"
    # )


def plot_ref_quad(ax, ref):
    # initially: plot full reference
    X_ref = ref[:, 0]
    Y_ref = ref[:, 1]
    Z_ref = ref[:, 2]
    ax.plot3D(X_ref, Y_ref, Z_ref, color="grey", label="reference")
    ax.set_xlim(np.min(X_ref), np.max(X_ref))
    ax.set_ylim(np.min(Y_ref), np.max(Y_ref))
    ax.set_zlim(np.min(Z_ref), np.max(Z_ref))
    ax.set_xlabel("x (in m)")
    ax.set_ylabel("y (in m)")
    ax.set_zlabel("z (in m)")
    set_axes_equal(ax)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    return ax


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def animate_quad(ref, trajectories, savefile=None, names=["APG"]):
    fig = plt.figure(figsize=(11, 10))
    ax = axes3d.Axes3D(fig)

    ax = plot_ref_quad(ax, ref)
    cols = ["blue", "red", "orange", "purple"]

    def update(i, ax, fig):
        ax.cla()
        ax = plot_ref_quad(ax, ref)
        for j, traj in enumerate(trajectories):
            ind = len(traj) - 1 if i >= len(traj) else i
            drone_col = "black" if len(names) == 1 else cols[j]
            ax = draw_quad(ax, traj[ind, :3], traj[ind, 3:6], c=drone_col)
            wframe = ax.plot3D(
                traj[:ind, 0],
                traj[:ind, 1],
                traj[:ind, 2],
                color=cols[j],
                label=names[j]
            )
        if len(names) > 1:
            ax.legend(bbox_to_anchor=(.9, .7))
        return wframe,

    dt = 0.05
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(len(ref)),
        fargs=(ax, fig),
        interval=dt * 1000
    )
    if savefile is not None:
        try:
            writervideo = animation.FFMpegWriter(fps=1 / dt, codec='libx264')
        except:
            writervideo = animation.FFMpegWriter(fps=1 / dt)
        anim.save(savefile, writer=writervideo)
        print("Video saved")
    plt.show()


def draw_fixed_wing(ax, position, euler, stretch=1, c="black"):
    back = position + body_to_world(euler, [-stretch * (7 / 6), 0, 0])
    # line from front to middle-front
    front = position + body_to_world(euler, [stretch * (5 / 6), 0, 0])
    front_middle = position + body_to_world(euler, [stretch * (1 / 3), 0, 0])
    right_back = position + body_to_world(euler, [-stretch * (1 / 6), 0.75, 0])
    left_back = position + body_to_world(euler, [-stretch * (1 / 6), -0.75, 0])
    middle_back = position + body_to_world(euler, [-stretch * (1 / 6), 0, 0])
    pointing_up = position + body_to_world(euler, [-stretch, 0, 0.25])
    left_up = position + body_to_world(euler, [-stretch, 0.25, 0.25])
    right_up = position + body_to_world(euler, [-stretch, -0.25, 0.25])
    # DRAW LINES
    # line from back to the front
    draw_line_3d(ax, back, front, color=c)
    # line from front middle to right back
    draw_line_3d(ax, front_middle, right_back, color=c)
    draw_line_3d(ax, front_middle, left_back, color=c)
    draw_line_3d(ax, middle_back, right_back, color=c)
    draw_line_3d(ax, middle_back, left_back, color=c)
    # small line pointing up in the back
    draw_line_3d(ax, back, pointing_up, color=c)
    # horizontal line on top
    draw_line_3d(ax, left_up, right_up, color=c)
    return ax


def plot_ref_wing(ax, target_point):
    # xlim is the maximum point
    ax.set_xlim(-1, target_point[-1, 0])
    ax.set_ylim(-7, 7)
    ax.set_zlim(-7, 7)
    ax.set_xlabel("x (in m)")
    ax.set_ylabel("y (in m)")
    ax.set_zlabel("z (in m)")
    s = ax.scatter3D(
        target_point[:, 0],
        target_point[:, 1],
        target_point[:, 2],
        marker="o",
        c="green",
        s=100,
        label="target point"
    )
    temp_target = np.concatenate((np.zeros((1, 3)), target_point))
    s = ax.plot3D(
        temp_target[:, 0],
        temp_target[:, 1],
        temp_target[:, 2],
        linestyle="--",
        c="grey",
        label="reference"
    )
    return ax


def animate_fixed_wing(
    target_point, trajectories, names=["APG"], savefile=None
):
    traj_lens = [len(t) for t in trajectories]
    dt = 0.05
    target_point = np.array(target_point)
    fig = plt.figure(figsize=(10, 10))
    ax = axes3d.Axes3D(fig)
    # set aspect ratio based on first one
    ax.set_box_aspect(
        (
            np.ptp(trajectories[0][:, 0] / 4), np.ptp(trajectories[0][:, 1]),
            np.ptp(trajectories[0][:, 2])
        )
    )
    cols = ["blue", "red", "orange", "purple"]
    ax = plot_ref_wing(ax, target_point)

    def update(i, ax, fig):
        ax.cla()
        ax = plot_ref_wing(ax, target_point)
        for j, traj in enumerate(trajectories):
            # if the trajectories have different lengths, stay at last position
            ind = len(traj) - 1 if i >= len(traj) else i
            euler = traj[ind, 6:9] * np.array([-1, 1, 1])
            drone_col = "black" if len(names) == 1 else cols[j]
            ax = draw_fixed_wing(ax, traj[ind, :3], euler, c=drone_col)
            ax.plot3D(
                traj[:ind, 0],
                traj[:ind, 1],
                traj[:ind, 2],
                color=cols[j],
                label=names[j]
            )
        if len(names) > 1:
            ax.legend(bbox_to_anchor=(.9, .7))

    ax.view_init(elev=20., azim=30)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=iter(range(max(traj_lens))),
        fargs=(ax, fig),
        interval=10
    )
    if savefile is not None:
        try:
            writervideo = animation.FFMpegWriter(fps=1 / dt, codec='libx264')
        except:
            writervideo = animation.FFMpegWriter(fps=1 / dt)
        anim.save(savefile, writer=writervideo)
        print("Video saved")
    plt.show()


if __name__ == "__main__":
    import os
    name = "wing"
    if name == "wing":
        target_point = [[50, 6, -4]]
        trajectories = []
        for model in ["mpc", "current_model", "ppo"]:
            trajectories.append(
                np.load(
                    os.path.join("output_video", f"wing_traj_{model}.npy")
                )
            )
        animate_fixed_wing(
            target_point,
            trajectories,
            names=["MPC", "APG", "PPO"],
            savefile="output_video/vid_comparison_wing.mp4"
        )
    else:
        ref = np.load(os.path.join("output_video", "quad_ref_mpc.npy"))
        trajectories = []
        for model in ["mpc", "current_model", "ppo"]:
            trajectories.append(
                np.load(
                    os.path.join("output_video", f"quad_traj_{model}.npy")
                )
            )
        animate_quad(
            ref,
            trajectories,
            names=["MPC", "APG", "PPO"],
            savefile="output_video/vid_comparison_quad.mp4"
        )
