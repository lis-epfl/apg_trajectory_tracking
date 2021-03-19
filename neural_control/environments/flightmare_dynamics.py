import torch
import numpy as np
from neural_control.environments.copter import copter_params
from neural_control.environments.drone_dynamics import (
    world_to_body_matrix, euler_rate
)
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

copter_params.inertia_J = torch.diag(inertia_vector)
copter_params.inertia_J_inv = torch.diag(1 / inertia_vector)
# torch.from_numpy(copter_params.frame_inertia
#                                              ).float().to(device)
kinv_ang_vel_tau = torch.diag(torch.tensor([16.6, 16.6, 5.0]).float())

# NEW STUFF:
t_BM_ = copter_params.arm_length * np.sqrt(0.5) * torch.tensor(
    [[1, -1, -1, 1], [-1, -1, 1, 1], [0, 0, 0, 0]]
)
kappa_ = 0.016  # rotor drag coefficient
motor_tau_inv_ = 1 / 0.05
b_allocation = torch.tensor(
    [[1, 1, 1, 1], t_BM_[0], t_BM_[1], kappa_ * torch.tensor([1, -1, 1, -1])]
)
b_allocation_inv = torch.inverse(b_allocation)

# ???
sim_dt = 0.02
max_t = 10.0

# other params
motor_tau = 0.0001
motor_tau_inv = 1 / motor_tau

thrust_map = torch.unsqueeze(
    torch.tensor(
        [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325]
    ), 1
)


def thrust_to_omega(thrusts):
    scale = 1.0 / (2.0 * thrust_map[0])
    offset = -thrust_map[1] * scale
    root = thrust_map[1]**2 - 4 * thrust_map[0] * (thrust_map[2] - thrusts)
    return offset + scale * torch.sqrt(root)


def motorOmegaToThrust(motor_omega_):
    motor_omega_ = torch.unsqueeze(motor_omega_, dim=2)
    omega_poly = torch.cat(
        (motor_omega_**2, motor_omega_, torch.ones(motor_omega_.size())),
        dim=2
    )
    return torch.matmul(omega_poly, thrust_map)


def run_motors(dt, motor_thrusts_des):
    # print("motor_thrusts_des\n", motor_thrusts_des.size())
    motor_omega = thrust_to_omega(motor_thrusts_des)

    # TODO clamp

    # TODO: actually the old motor thrust is taken into account here,
    # but the factor c is super low (hoch -218)
    # so it doesn't actually matter, so I left it out
    # motor step response
    # scalar_c = torch.exp(-dt * motor_tau_inv)
    # motor_omega_ = scalar_c * motor_omega_prev + (1.0 - c) * motor_omega_des

    # convert back
    motor_thrusts = motorOmegaToThrust(motor_omega)
    # print("motor_thrusts\n", motor_thrusts.size())
    # TODO: clamp?
    return motor_thrusts[:, :, 0]


def linear_dynamics(force, attitude, velocity):
    m = copter_params.mass

    world_to_body = world_to_body_matrix(attitude)
    body_to_world = torch.transpose(world_to_body, 1, 2)

    # print("force in ld ", force.size())
    thrust = 1 / m * torch.matmul(body_to_world, torch.unsqueeze(force, 2))
    # print("thrust", thrust.size())
    # drag = velocity * TODO: dynamics.drag_coeff??
    thrust_min_grav = thrust[:, :, 0] + copter_params.gravity
    return thrust_min_grav  # - drag


def run_flight_control(thrust, av, body_rates, cross_prod):
    """
    thrust: command first signal (around 9.81)
    omega = av: current angular velocity
    command = body_rates: body rates in command
    """
    force = torch.unsqueeze(copter_params.mass * thrust, 1)

    # constants
    omega_change = torch.unsqueeze(body_rates - av, 2)
    kinv_times_change = torch.matmul(kinv_ang_vel_tau, omega_change)
    first_part = torch.matmul(copter_params.inertia_J, kinv_times_change)
    # print("first_part", first_part.size())
    body_torque_des = first_part[:, :, 0] + cross_prod

    thrust_and_torque = torch.unsqueeze(
        torch.cat((force, body_torque_des), dim=1), 2
    )
    # print(thrust_and_torque.size())
    motor_thrusts_des = torch.matmul(b_allocation_inv, thrust_and_torque)
    # TODO: clamp?

    return motor_thrusts_des[:, :, 0]


def pretty_print(varname, torch_var):
    np.set_printoptions(suppress=1, precision=7)
    if len(torch_var) > 1:
        print("ERR: batch size larger 1", torch_var.size())
    print(varname, torch_var[0].detach().numpy())


def flightmare_dynamics_function(action, state, dt):
    """
    Pytorch implementation of the dynamics in Flightmare simulator
    """
    # extract state
    position = state[:, :3]
    attitude = state[:, 3:6]
    velocity = state[:, 6:9]
    angular_velocity = state[:, 9:]

    # action is normalized between 0 and 1 --> rescale
    total_thrust = action[:, 0] * 15 - 7.5 + 9.81
    body_rates = action[:, 1:] - .5

    # ctl_dt ist simulation time,
    # remainer wird immer -sim_dt gemacht in jedem loop

    # precompute cross product
    inertia_av = torch.matmul(
        copter_params.inertia_J, torch.unsqueeze(angular_velocity, 2)
    )[:, :, 0]
    cross_prod = torch.cross(angular_velocity, inertia_av, dim=1)

    motor_thrusts_des = run_flight_control(
        total_thrust, angular_velocity, body_rates, cross_prod
    )
    motor_thrusts = run_motors(dt, motor_thrusts_des)

    force_torques = torch.matmul(
        b_allocation, torch.unsqueeze(motor_thrusts, 2)
    )[:, :, 0]

    # 1) linear dynamics
    force_expanded = torch.unsqueeze(force_torques[:, 0], 1)
    f_s = force_expanded.size()
    force = torch.cat(
        (torch.zeros(f_s), torch.zeros(f_s), force_expanded), dim=1
    )

    acceleration = linear_dynamics(force, attitude, velocity)

    position = position + 0.5 * dt * dt * acceleration + 0.5 * dt * velocity
    velocity = velocity + dt * acceleration

    # 2) angular acceleration
    tau = force_torques[:, 1:]
    angular_acc = torch.matmul(
        copter_params.inertia_J_inv, torch.unsqueeze((tau - cross_prod), 2)
    )[:, :, 0]
    new_angular_velocity = angular_velocity + dt * angular_acc

    # other option: use quaternion
    # --> also slight error to flightmare, even when using euler, no idea why
    # from neural_control.utils.q_funcs import (
    #     q_dot_new, euler_to_quaternion, quaternion_to_euler
    # )
    # quaternion = euler_to_quaternion(
    #     attitude[0, 0].item(), attitude[0, 1].item(), attitude[0, 2].item()
    # )
    # print("quaternion", quaternion)
    # np.set_printoptions(suppress=1, precision=7)
    # av_test = angular_velocity[0].numpy()
    # quaternion_omega = np.array([av_test[0], av_test[1], av_test[2]])
    # print("quaternion_omega", quaternion_omega)
    # q_dot = q_dot_new(quaternion, quaternion_omega)
    # print("q dot", q_dot)
    # # integrate
    # new_quaternion = quaternion + dt * q_dot
    # print("new_quaternion", new_quaternion)
    # new_quaternion = new_quaternion / np.linalg.norm(new_quaternion)
    # print("new_quaternion", new_quaternion)
    # new_euler = quaternion_to_euler(new_quaternion)
    # print("new euler", new_euler)

    # pretty_print("attitude before", attitude)

    attitude = attitude + dt * euler_rate(attitude, angular_velocity)

    # set final state
    state = torch.hstack((position, attitude, velocity, new_angular_velocity))
    return state.float()


if __name__ == "__main__":
    action = torch.tensor([[0.45, 0.46, 0.3, 0.6]])

    state = [
        -0.203302, -8.12219, 0.484883, -0.15613, -0.446313, 0.25728, -4.70952,
        0.627684, -2.506545, -0.039999, -0.200001, 0.1
    ]
    # state = [2, 3, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    new_state = flightmare_dynamics_function(
        action, torch.tensor([state]), 0.05
    )
