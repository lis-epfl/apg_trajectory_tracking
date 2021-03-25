import torch
import numpy as np
from neural_control.environments.dynamics import Dynamics
import casadi as ca


class FlightmareDynamics(Dynamics):

    def __init__(self, modified_params={}, simulate_rotors=False):
        super().__init__(modified_params=modified_params)
        # new parameters needed for flightmare simulation
        self.t_BM_ = self.arm_length * np.sqrt(0.5) * torch.tensor(
            [[1, -1, -1, 1], [-1, -1, 1, 1], [0, 0, 0, 0]]
        )
        self.kappa_ = 0.016  # rotor drag coefficient
        self.motor_tau_inv_ = 1 / 0.05
        self.b_allocation = torch.tensor(
            [
                [1, 1, 1, 1], self.t_BM_[0], self.t_BM_[1],
                self.kappa_ * torch.tensor([1, -1, 1, -1])
            ]
        )
        self.b_allocation_inv = torch.inverse(self.b_allocation)

        # other params
        self.motor_tau = 0.0001
        self.motor_tau_inv = 1 / self.motor_tau
        self.thrust_map = torch.unsqueeze(
            torch.tensor(
                [
                    1.3298253500372892e-06, 0.0038360810526746033,
                    -1.7689986848125325
                ]
            ), 1
        )

    def thrust_to_omega(self, thrusts):
        scale = 1.0 / (2.0 * self.thrust_map[0])
        offset = -self.thrust_map[1] * scale
        root = self.thrust_map[
            1]**2 - 4 * self.thrust_map[0] * (self.thrust_map[2] - thrusts)
        return offset + scale * torch.sqrt(root)

    def motorOmegaToThrust(self, motor_omega_):
        motor_omega_ = torch.unsqueeze(motor_omega_, dim=2)
        omega_poly = torch.cat(
            (motor_omega_**2, motor_omega_, torch.ones(motor_omega_.size())),
            dim=2
        )
        return torch.matmul(omega_poly, self.thrust_map)

    def run_motors(self, dt, motor_thrusts_des):
        # print("motor_thrusts_des\n", motor_thrusts_des.size())
        motor_omega = self.thrust_to_omega(motor_thrusts_des)

        # TODO clamp

        # TODO: actually the old motor thrust is taken into account here,
        # but the factor c is super low (hoch -218)
        # so it doesn't actually matter, so I left it out
        # motor step response
        # scalar_c = torch.exp(-dt * motor_tau_inv)
        # motor_omega_ = scalar_c * motor_omega_prev + (1.0 - c) * motor_omega_des

        # convert back
        motor_thrusts = self.motorOmegaToThrust(motor_omega)
        # print("motor_thrusts\n", motor_thrusts.size())
        # TODO: clamp?
        return motor_thrusts[:, :, 0]

    def linear_dynamics(self, force, attitude, velocity):
        """
        linear dynamics
        no drag so far
        """

        world_to_body = self.world_to_body_matrix(attitude)
        body_to_world = torch.transpose(world_to_body, 1, 2)

        # print("force in ld ", force.size())
        thrust = self.copter_params.down_drag * 1 / self.mass * torch.matmul(
            body_to_world, torch.unsqueeze(force, 2)
        )
        # print("thrust", thrust.size())
        # drag = velocity * TODO: dynamics.drag_coeff??
        thrust_min_grav = thrust[:, :, 0] + self.torch_gravity
        return thrust_min_grav  # - drag

    def run_flight_control(self, thrust, av, body_rates, cross_prod):
        """
        thrust: command first signal (around 9.81)
        omega = av: current angular velocity
        command = body_rates: body rates in command
        """
        force = torch.unsqueeze(self.mass * thrust, 1)

        # constants
        omega_change = torch.unsqueeze(body_rates - av, 2)
        kinv_times_change = torch.matmul(
            self.torch_kinv_ang_vel_tau, omega_change
        )
        first_part = torch.matmul(self.torch_inertia_J, kinv_times_change)
        # print("first_part", first_part.size())
        body_torque_des = first_part[:, :, 0] + cross_prod

        thrust_and_torque = torch.unsqueeze(
            torch.cat((force, body_torque_des), dim=1), 2
        )
        return thrust_and_torque[:, :, 0]

    def _pretty_print(self, varname, torch_var):
        np.set_printoptions(suppress=1, precision=7)
        if len(torch_var) > 1:
            print("ERR: batch size larger 1", torch_var.size())
        print(varname, torch_var[0].detach().numpy())

    def simulate_quadrotor(self, action, state, dt):
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
            self.torch_inertia_J, torch.unsqueeze(angular_velocity, 2)
        )[:, :, 0]
        cross_prod = torch.cross(angular_velocity, inertia_av, dim=1)

        force_torques = self.run_flight_control(
            total_thrust, angular_velocity, body_rates, cross_prod
        )
        # # SIMULATE ROTORS
        # motor_thrusts_des = torch.matmul(b_allocation_inv,
        #            thrust_and_torque)[:, :, 0]
        # # TODO: clamp?
        # motor_thrusts = self.run_motors(dt, motor_thrusts_des)
        # force_torques = torch.matmul(
        #     b_allocation, torch.unsqueeze(motor_thrusts, 2)
        # )[:, :, 0]

        # 1) linear dynamics
        force_expanded = torch.unsqueeze(force_torques[:, 0], 1)
        f_s = force_expanded.size()
        force = torch.cat(
            (torch.zeros(f_s), torch.zeros(f_s), force_expanded), dim=1
        )

        acceleration = self.linear_dynamics(force, attitude, velocity)

        position = (
            position + 0.5 * dt * dt * acceleration + 0.5 * dt * velocity
        )
        velocity = velocity + dt * acceleration

        # 2) angular acceleration
        tau = force_torques[:, 1:]
        angular_acc = torch.matmul(
            self.torch_inertia_J_inv, torch.unsqueeze((tau - cross_prod), 2)
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

        attitude = attitude + dt * self.euler_rate(attitude, angular_velocity)

        # set final state
        state = torch.hstack(
            (position, attitude, velocity, new_angular_velocity)
        )
        return state.float()


class FlightmareDynamicsMPC(Dynamics):

    def __init__(self):
        super().__init__()

        # TODO: run rotors params:
        # kappa_ = 0.016
        # motor_tau_inv_ = 1 / 0.05
        # b_allocation = ca.SX(self.b_allocation_np)
        # b_allocation_inv = ca.SX(np.linalg.inv(self.b_allocation_np))
        # motor_tau = 0.0001
        # motor_tau_inv = 1 / motor_tau

    def drone_dynamics_flightmare(self, dt):

        # # # # # # # # # # # # # # # # # # #
        # --------- State ------------
        # # # # # # # # # # # # # # # # # # #

        # position
        px, py, pz = ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz')
        # attitude
        ax, ay, az = ca.SX.sym('ax'), ca.SX.sym('ay'), ca.SX.sym('az')
        # vel
        vx, vy, vz = ca.SX.sym('vx'), ca.SX.sym('vy'), ca.SX.sym('vz')
        # angular velocity
        avx, avy, avz = ca.SX.sym('avx'), ca.SX.sym('avy'), ca.SX.sym('avz')

        # -- conctenated vector
        self._x = ca.vertcat(px, py, pz, ax, ay, az, vx, vy, vz, avx, avy, avz)

        # # # # # # # # # # # # # # # # # # #
        # --------- Control Command ------------
        # # # # # # # # # # # # # # # # # # #

        thrust, wx, wy, wz = ca.SX.sym('thrust'), ca.SX.sym('wx'), \
            ca.SX.sym('wy'), ca.SX.sym('wz')

        # -- conctenated vector
        self._u = ca.vertcat(thrust, wx, wy, wz)

        force = thrust * 15 - 7.5 + 9.81
        body_rates = ca.vertcat(wx - .5, wy - .5, wz - .5)

        # compute cross product (needed at two steps)
        av = ca.vertcat(avx, avy, avz)
        inertia_times_av = self.ca_inertia_vector * av
        cross_prod = ca.cross(av, inertia_times_av)

        # run flight control
        # force = thrust_scaled # TODO: why not using mass????
        # action to body torques
        omega_change = body_rates - av
        first_part = (
            self.ca_inertia_vector * self.ca_kinv_ang_vel_tau * omega_change
        )
        body_torques = first_part + cross_prod

        # TODO simulate rotors?
        # thrust_and_torque = ca.vertcat(force, body_torques)

        # linear dynamics
        Cy = ca.cos(az)
        Sy = ca.sin(az)
        Cp = ca.cos(ay)
        Sp = ca.sin(ay)
        Cr = ca.cos(ax)
        Sr = ca.sin(ax)

        acc_x = (Cy * Sp * Cr + Sr * Sy) * force
        acc_y = (Cr * Sy * Sp - Cy * Sr) * force
        acc_z = (Cr * Cp) * force - 9.81

        px_new = px + 0.5 * dt * dt * acc_x + 0.5 * dt * vx
        py_new = py + 0.5 * dt * dt * acc_y + 0.5 * dt * vy
        pz_new = pz + 0.5 * dt * dt * acc_z + 0.5 * dt * vz
        vx_new = vx + dt * acc_x
        vy_new = vy + dt * acc_y
        vz_new = vz + dt * acc_z

        # angular dynamics
        angular_acc = self.ca_inertia_vector_inv * (body_torques - cross_prod)
        # angular_velocity = angular_velocity + dt * angular_acc
        avx_new = avx + dt * angular_acc[0]
        avy_new = avy + dt * angular_acc[1]
        avz_new = avz + dt * angular_acc[2]

        # attitude = attitude + dt * euler_rate(attitude, new angular_velocity)
        euler_rate_x = avx - ca.sin(ay) * avz
        euler_rate_y = ca.cos(ax) * avy + ca.cos(ay) * ca.sin(ax) * avz
        euler_rate_z = -ca.sin(ax) * avy + ca.cos(ay) * ca.cos(ax) * avz
        ax_new = ax + dt * euler_rate_x
        ay_new = ay + dt * euler_rate_y
        az_new = az + dt * euler_rate_z

        # stack together
        X = ca.vertcat(
            px_new, py_new, pz_new, ax_new, ay_new, az_new, vx_new, vy_new,
            vz_new, avx_new, avy_new, avz_new
        )
        # Fold
        F = ca.Function('F', [self._x, self._u], [X], ['x', 'u'], ['ode'])
        return F


if __name__ == "__main__":
    action = [0.45, 0.46, 0.3, 0.6]

    state = [
        -0.203302, -8.12219, 0.484883, -0.15613, -0.446313, 0.25728, -4.70952,
        0.627684, -2.506545, -0.039999, -0.200001, 0.1
    ]
    # state = [2, 3, 4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    dyn = FlightmareDynamics()
    new_state = dyn.simulate_quadrotor(
        torch.tensor([action]), torch.tensor([state]), 0.05
    )
    print("new state flightmare", new_state)

    dyn_mpc = FlightmareDynamicsMPC()
    F = dyn_mpc.drone_dynamics_flightmare(0.05)
    new_state_mpc = F(np.array(state), np.array(action))
    print("new state mpc", new_state_mpc)
