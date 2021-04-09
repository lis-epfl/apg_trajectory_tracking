import casadi as ca
import torch
import numpy as np
from types import SimpleNamespace


class Dynamics:

    def __init__(
        self,
        thrust_factor=5.723e-06,
        drag_factor=1.717e-07,
        mass=0.723,
        rotational_drag=np.array([0., 0., 0.]),
        translational_drag=np.array([0, 0, 0]),
        arm_length=0.31,
        rotor_inertia=7.321e-05,
        frame_inertia=np.array([4.5, 4.5, 7.]),
        gravity=np.array([0., 0., -9.81]),
        max_rotor_speed=1000.0,
        rotor_speed_half_time=0.0625,
        kinv_ang_vel_tau=np.array([16.6, 16.6, 5.])
    ):
        device = "cpu"
        # NUMPY PARAMETERS
        self.mass = mass
        self.arm_length = arm_length
        self.thrust_factor = thrust_factor
        self.drag_factor = drag_factor
        self.rotor_inertia = rotor_inertia
        self.max_rotor_speed = max_rotor_speed
        self.kinv_ang_vel_tau = kinv_ang_vel_tau
        self.inertia_vector = (mass / 12.0 * arm_length**2 * frame_inertia)

        # TORCH PARAMETERS
        # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.copter_params = SimpleNamespace(**self.copter_params)
        self.torch_translational_drag = torch.from_numpy(translational_drag
                                                         ).float().to(device)
        self.torch_gravity = torch.from_numpy(gravity).to(device)
        self.torch_rotational_drag = torch.from_numpy(rotational_drag
                                                      ).float().to(device)
        self.torch_inertia_vector = torch.from_numpy(self.inertia_vector
                                                     ).float().to(device)

        self.torch_inertia_J = torch.diag(self.torch_inertia_vector)
        self.torch_inertia_J_inv = torch.diag(1 / self.torch_inertia_vector)
        self.torch_kinv_vector = torch.tensor(kinv_ang_vel_tau).float()
        self.torch_kinv_ang_vel_tau = torch.diag(self.torch_kinv_vector)

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
