import torch
import numpy as np
from neural_control.environments.copter import copter_params
from types import SimpleNamespace

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
copter_params = SimpleNamespace(**copter_params)
copter_params.translational_drag = torch.from_numpy(
    copter_params.translational_drag
).to(device)
copter_params.gravity = torch.from_numpy(copter_params.gravity).to(device)
copter_params.rotational_drag = torch.from_numpy(
    copter_params.rotational_drag
).to(device)
# estimate intertia as in flightmare
inertia_vector = (
    copter_params.mass / 12.0 * copter_params.arm_length**2 *
    torch.tensor([4.5, 4.5, 7])
).float().to(device)
copter_params.frame_inertia = torch.diag(inertia_vector)
# torch.from_numpy(copter_params.frame_inertia
#                                              ).float().to(device)
kinv_ang_vel_tau = torch.diag(torch.tensor([16.6, 16.6, 5.0]).float())


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


def linear_dynamics(squared_rotor_speed, attitude, velocity):
    """
    Calculates the linear acceleration of a quadcopter with parameters
    `copter_params` that is currently in the dynamics state composed of:
    :param rotor_speed: current rotor speeds
    :param attitude: current attitude
    :param velocity: current velocity
    :return: Linear acceleration in world frame.
    """
    m = copter_params.mass
    b = copter_params.thrust_factor
    Kt = copter_params.translational_drag

    world_to_body = world_to_body_matrix(attitude)
    body_to_world = torch.transpose(world_to_body, 1, 2)

    constant_vec = torch.zeros(3).to(device)
    constant_vec[2] = 1

    thrust = 1 / m * torch.mul(
        torch.matmul(body_to_world, constant_vec).t(), squared_rotor_speed
    ).t()
    Ktw = torch.matmul(
        body_to_world, torch.matmul(torch.diag(Kt).float(), world_to_body)
    )
    # drag = torch.squeeze(torch.matmul(Ktw, torch.unsqueeze(velocity, 2)) / m)
    thrust_minus_drag = thrust + copter_params.gravity
    # version for batch size 1 (working version)
    # summed = torch.add(
    #     torch.transpose(drag * (-1), 0, 1), thrust
    # ) + copter_params.gravity
    # print("output linear", thrust_minus_drag.size())
    return thrust_minus_drag


def to_euler_matrix(attitude):
    # attitude is [roll, pitch, yaw]
    pitch = attitude[:, 1]
    roll = attitude[:, 0]
    Cp = torch.cos(pitch)
    Sp = torch.sin(pitch)
    Cr = torch.cos(roll)
    Sr = torch.sin(roll)

    zero_vec_bs = torch.zeros(Sp.size()).to(device)
    ones_vec_bs = torch.ones(Sp.size()).to(device)

    # create matrix
    m1 = torch.transpose(torch.vstack([ones_vec_bs, zero_vec_bs, -Sp]), 0, 1)
    m2 = torch.transpose(torch.vstack([zero_vec_bs, Cr, Cp * Sr]), 0, 1)
    m3 = torch.transpose(torch.vstack([zero_vec_bs, -Sr, Cp * Cr]), 0, 1)
    matrix = torch.stack((m1, m2, m3), dim=1)

    # matrix = torch.tensor([[1, 0, -Sp], [0, Cr, Cp * Sr], [0, -Sr, Cp * Cr]])
    return matrix


def euler_rate(attitude, angular_velocity):
    euler_matrix = to_euler_matrix(attitude)
    together = torch.matmul(
        euler_matrix, torch.unsqueeze(angular_velocity.float(), 2)
    )
    # print("output euler rate", together.size())
    return torch.squeeze(together)


def action_to_body_torques(av, body_rates):
    """
    omega is current angular velocity
    thrust, body_rates: current command
    """
    # constants
    omega_change = torch.unsqueeze(body_rates - av, 2)
    kinv_times_change = torch.matmul(kinv_ang_vel_tau, omega_change)
    first_part = torch.matmul(copter_params.frame_inertia, kinv_times_change)
    # print("first_part", first_part.size())
    inertia_av = torch.matmul(
        copter_params.frame_inertia, torch.unsqueeze(av, 2)
    )[:, :, 0]
    # print(inertia_av.size())
    second_part = torch.cross(av, inertia_av, dim=1)
    # print("second_part", second_part.size())
    body_torque_des = first_part[:, :, 0] + second_part
    # print("body_torque_des", body_torque_des.size())
    return body_torque_des


def simulate_quadrotor(action, state, dt=0.02):
    """
    Simulate the dynamics of the quadrotor for the timestep given
    in `dt`. First the rotor speeds are updated according to the desired
    rotor speed, and then linear and angular accelerations are calculated
    and integrated.
    Arguments:
        action: float tensor of size (BATCH_SIZE, 4) - rotor thrust
        state: float tensor of size (BATCH_SIZE, 16) - drone state (see below)
    Returns:
        Next drone state (same size as state)
    """
    # extract state
    position = state[:, :3]
    attitude = state[:, 3:6]
    velocity = state[:, 6:9]
    angular_velocity = state[:, 9:]

    # action is normalized between 0 and 1 --> rescale
    total_thrust = action[:, 0] * 10 - 5 + 7
    body_rates = action[:, 1:] - .5

    acceleration = linear_dynamics(total_thrust, attitude, velocity)

    ang_momentum = action_to_body_torques(angular_velocity, body_rates)
    # angular_momentum_body_frame(rotor_speed, angular_velocity)
    angular_acc = ang_momentum / inertia_vector
    # update state variables
    position = position + 0.5 * dt * dt * acceleration + 0.5 * dt * velocity
    velocity = velocity + dt * acceleration
    angular_velocity = angular_velocity + dt * angular_acc
    attitude = attitude + dt * euler_rate(attitude, angular_velocity)
    # set final state
    state = torch.hstack((position, attitude, velocity, angular_velocity))
    return state.float()
