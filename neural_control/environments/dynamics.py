import casadi as ca
import torch
import numpy as np
from types import SimpleNamespace


class Dynamics:

    def __init__(self, modified_params={}):
        dict_copter_params = {
            "thrust_factor": 5.723e-6,
            "drag_factor": 1.717e-7,
            "mass": 0.723,
            "rotational_drag": np.array([1, 1, 1]) * 1e-4,
            "translational_drag": np.array([1, 1, 1]) * 1e-4,
            "arm_length": 0.31,
            "rotor_inertia": 7.321e-5,
            "frame_inertia": np.array([4.5, 4.5, 7]),
            "gravity": np.array([0.0, 0.0, -9.81]),
            "max_rotor_speed": 1000.0,
            "rotor_speed_half_time": 1.0 / 16,
            "kinv_ang_vel_tau": np.array([16.6, 16.6, 5.0]),
            "down_drag": 1
        }
        # change the parameters that should be motified
        dict_copter_params.update(modified_params)
        self.copter_params = SimpleNamespace(**dict_copter_params)
        device = "cpu"
        # NUMPY PARAMETERS
        self.mass = self.copter_params.mass
        self.arm_length = self.copter_params.arm_length
        self.thrust_factor = self.copter_params.thrust_factor
        self.drag_factor = self.copter_params.drag_factor
        self.rotor_inertia = self.copter_params.rotor_inertia
        self.max_rotor_speed = self.copter_params.max_rotor_speed
        self.kinv_ang_vel_tau = self.copter_params.kinv_ang_vel_tau
        self.inertia_vector = (
            self.copter_params.mass / 12.0 * self.copter_params.arm_length**2 *
            self.copter_params.frame_inertia
        )

        # TORCH PARAMETERS
        # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.copter_params = SimpleNamespace(**self.copter_params)
        self.torch_translational_drag = torch.from_numpy(
            self.copter_params.translational_drag
        ).to(device)
        self.torch_gravity = torch.from_numpy(self.copter_params.gravity
                                              ).to(device)
        self.torch_rotational_drag = torch.from_numpy(
            self.copter_params.rotational_drag
        ).to(device)
        self.torch_inertia_vector = torch.from_numpy(self.inertia_vector
                                                     ).float().to(device)

        self.torch_inertia_J = torch.diag(self.torch_inertia_vector)
        self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector)
        self.torch_kinv_ang_vel_tau = torch.diag(
            torch.tensor(self.copter_params.kinv_ang_vel_tau).float()
        )

        # CASADI PARAMETERS
        self.ca_inertia_vector = ca.SX(self.inertia_vector)
        self.ca_inertia_vector_inv = ca.SX(1 / self.inertia_vector)
        self.ca_kinv_ang_vel_tau = ca.SX(self.kinv_ang_vel_tau)

    @staticmethod
    def world_to_body_matrix(attitude):
        """
        Creates a transformation matrix for directions from world frame
        to body frame for a body with attitude given by `euler` Euler angles.
        :param euler: The Euler angles of the body frame.
        :return: The transformation matrix.
        """

        # check if we have a cached result already available
        roll = attitude[:, 0]
        pitch = attitude[:, 1]
        yaw = attitude[:, 2]

        Cy = torch.cos(yaw)
        Sy = torch.sin(yaw)
        Cp = torch.cos(pitch)
        Sp = torch.sin(pitch)
        Cr = torch.cos(roll)
        Sr = torch.sin(roll)

        # create matrix
        m1 = torch.transpose(torch.vstack([Cy * Cp, Sy * Cp, -Sp]), 0, 1)
        m2 = torch.transpose(
            torch.vstack(
                [Cy * Sp * Sr - Cr * Sy, Cr * Cy + Sr * Sy * Sp, Cp * Sr]
            ), 0, 1
        )
        m3 = torch.transpose(
            torch.vstack(
                [Cy * Sp * Cr + Sr * Sy, Cr * Sy * Sp - Cy * Sr, Cr * Cp]
            ), 0, 1
        )
        matrix = torch.stack((m1, m2, m3), dim=1)

        return matrix

    @staticmethod
    def to_euler_matrix(attitude):
        # attitude is [roll, pitch, yaw]
        pitch = attitude[:, 1]
        roll = attitude[:, 0]
        Cp = torch.cos(pitch)
        Sp = torch.sin(pitch)
        Cr = torch.cos(roll)
        Sr = torch.sin(roll)

        zero_vec_bs = torch.zeros(Sp.size())
        ones_vec_bs = torch.ones(Sp.size())

        # create matrix
        m1 = torch.transpose(
            torch.vstack([ones_vec_bs, zero_vec_bs, -Sp]), 0, 1
        )
        m2 = torch.transpose(torch.vstack([zero_vec_bs, Cr, Cp * Sr]), 0, 1)
        m3 = torch.transpose(torch.vstack([zero_vec_bs, -Sr, Cp * Cr]), 0, 1)
        matrix = torch.stack((m1, m2, m3), dim=1)

        # matrix = torch.tensor([[1, 0, -Sp], [0, Cr, Cp * Sr], [0, -Sr, Cp * Cr]])
        return matrix

    @staticmethod
    def euler_rate(attitude, angular_velocity):
        euler_matrix = Dynamics.to_euler_matrix(attitude)
        together = torch.matmul(
            euler_matrix, torch.unsqueeze(angular_velocity.float(), 2)
        )
        # print("output euler rate", together.size())
        return torch.squeeze(together)
