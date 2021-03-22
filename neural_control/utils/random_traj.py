import numpy as np
from .generate_trajectory import generate_trajectory


class Random:

    def __init__(
        self,
        drone_state,
        render=False,
        renderer=None,
        speed_factor=.6,
        horizon=10,
        duration=None,
        dt=0.05,
        **kwargs
    ):
        """
        Create random trajectory
        """
        self.horizon = horizon
        self.dt = dt
        # make variable whether we are already finished with the trajectory
        self.finished = False
        if render and renderer is None:
            raise ValueError("if render is true, need to input renderer")

        if duration is None:
            duration = 10 / 0.05 * dt
        points_3d = generate_trajectory(
            duration, dt, speed_factor=speed_factor
        )
        self.full_ref = points_3d
        self.initial_pos = points_3d[0, :3]
        # all_training_data = np.load("training_data.npy")
        # rand_ind = np.random.randint(0, len(all_training_data) // 501, 1)
        # start = int(rand_ind * 501)
        # points_3d = all_training_data[start:start + 501]

        # subtract current position to start there
        # print(drone_state[:3], points_3d[:3])
        # points_3d[:, :3] = points_3d[:, :3] - points_3d[
        #     0, :3] + drone_state[:3]
        # TODO merge: without it it looks nicer, but this was in merge

        self.reference = points_3d[:, :6]
        # np.zeros((len(points_3d), 9))
        # self.reference[:, :3] = points_3d[:, :3]
        # self.reference[:, 3:6] = points_3d[:, 6:9]

        self.ref_len = len(self.reference)
        self.target_ind = 0
        self.current_ind = 0

        # draw trajectory on renderer
        if render:
            renderer.add_object(PolyObject(self.reference))

    def get_ref_traj(self, drone_state, drone_acc):
        """
        Given the current position, compute a min snap trajectory to the next
        target
        """
        # if already at end, return zero velocities and accelerations
        if self.current_ind >= len(self.reference) - self.horizon:
            zero_ref = np.zeros(
                (
                    self.horizon - (self.ref_len - self.current_ind),
                    self.reference.shape[1]
                )
            )
            zero_ref[:, :3] = self.reference[-1, :3]
            left_over_ref = self.reference[self.current_ind:]
            return np.vstack((left_over_ref, zero_ref))
        out_ref = self.reference[self.current_ind + 1:self.current_ind +
                                 self.horizon + 1]
        self.current_ind += 1
        return out_ref

    def project_on_ref(self, drone_state):
        """
        Project drone state onto the trajectory
        """
        return self.reference[self.current_ind, :3]

    def get_current_full_state(self):
        pos_vel = self.full_ref[self.current_ind]
        return np.hstack((pos_vel[:3], pos_vel[6:], pos_vel[3:6], np.zeros(3)))


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