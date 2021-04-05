import torch
import numpy as np
from neural_control.dynamics.quad_dynamics_base import Dynamics
import casadi as ca


class SimpleDynamics(Dynamics):

    def linear_dynamics(self, squared_rotor_speed, attitude, velocity):
        """
        Calculates the linear acceleration of a quadcopter with parameters
        `copter_params` that is currently in the dynamics state composed of:
        :param rotor_speed: current rotor speeds
        :param attitude: current attitude
        :param velocity: current velocity
        :return: Linear acceleration in world frame.
        """
        m = self.mass
        b = self.thrust_factor
        Kt = self.torch_translational_drag

        world_to_body = self.world_to_body_matrix(attitude)
        body_to_world = torch.transpose(world_to_body, 1, 2)

        constant_vec = torch.zeros(3)
        constant_vec[2] = 1

        thrust = 1 / m * torch.mul(
            torch.matmul(body_to_world, constant_vec).t(), squared_rotor_speed
        ).t()
        Ktw = torch.matmul(
            body_to_world, torch.matmul(torch.diag(Kt).float(), world_to_body)
        )
        # drag = torch.squeeze(torch.matmul(Ktw, torch.unsqueeze(velocity, 2)) / m)
        thrust_minus_drag = thrust + self.torch_gravity
        # version for batch size 1 (working version)
        # summed = torch.add(
        #     torch.transpose(drag * (-1), 0, 1), thrust
        # ) + copter_params.gravity
        # print("output linear", thrust_minus_drag.size())
        return thrust_minus_drag

    def action_to_body_torques(self, av, body_rates):
        """
        omega is current angular velocity
        thrust, body_rates: current command
        """
        # constants
        omega_change = torch.unsqueeze(body_rates - av, 2)
        kinv_times_change = torch.matmul(
            self.torch_kinv_ang_vel_tau, omega_change
        )
        first_part = torch.matmul(self.torch_inertia_J, kinv_times_change)
        # print("first_part", first_part.size())
        inertia_av = torch.matmul(
            self.torch_inertia_J, torch.unsqueeze(av, 2)
        )[:, :, 0]
        # print(inertia_av.size())
        second_part = torch.cross(av, inertia_av, dim=1)
        # print("second_part", second_part.size())
        body_torque_des = first_part[:, :, 0] + second_part
        # print("body_torque_des", body_torque_des.size())
        return body_torque_des

    def simulate_quadrotor(self, action, state, dt=0.02):
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
        total_thrust = action[:, 0] * 15 - 7.5 + 9.81
        body_rates = action[:, 1:] - .5

        acceleration = self.linear_dynamics(total_thrust, attitude, velocity)

        ang_momentum = self.action_to_body_torques(
            angular_velocity, body_rates
        )
        # angular_momentum_body_frame(rotor_speed, angular_velocity)
        angular_acc = ang_momentum / self.torch_inertia_vector
        # update state variables
        position = position + 0.5 * dt * dt * acceleration + 0.5 * dt * velocity
        velocity = velocity + dt * acceleration
        angular_velocity = angular_velocity + dt * angular_acc
        attitude = attitude + dt * self.euler_rate(attitude, angular_velocity)
        # set final state
        state = torch.hstack((position, attitude, velocity, angular_velocity))
        return state.float()


class SimpleDynamicsMPC(Dynamics):

    def drone_dynamics_simple(self, dt):
        """
        Dynamics function in casadi for MPC optimization
        """
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

        thrust_scaled = thrust * 15 - 7.5 + 9.81

        # linear dynamics
        Cy = ca.cos(az)
        Sy = ca.sin(az)
        Cp = ca.cos(ay)
        Sp = ca.sin(ay)
        Cr = ca.cos(ax)
        Sr = ca.sin(ax)

        const = thrust_scaled / self.mass
        acc_x = (Cy * Sp * Cr + Sr * Sy) * const
        acc_y = (Cr * Sy * Sp - Cy * Sr) * const
        acc_z = (Cr * Cp) * const - 9.81

        px_new = px + 0.5 * dt * dt * acc_x + 0.5 * dt * vx
        py_new = py + 0.5 * dt * dt * acc_y + 0.5 * dt * vy
        pz_new = pz + 0.5 * dt * dt * acc_z + 0.5 * dt * vz
        vx_new = vx + dt * acc_x
        vy_new = vy + dt * acc_y
        vz_new = vz + dt * acc_z

        # angular dynamics

        body_rates = ca.vertcat(wx - .5, wy - .5, wz - .5)
        av = ca.vertcat(avx, avy, avz)

        # action to body torques
        omega_change = body_rates - av
        first_part = (
            self.ca_inertia_vector * self.ca_kinv_ang_vel_tau * omega_change
        )
        inertia_times_av = self.ca_inertia_vector * av
        second_part = ca.cross(av, inertia_times_av)  # dim??
        body_torques = (first_part + second_part) / self.ca_inertia_vector

        # angular_velocity = angular_velocity + dt * angular_acc
        avx_new = avx + dt * body_torques[0]
        avy_new = avy + dt * body_torques[1]
        avz_new = avz + dt * body_torques[2]

        # attitude = attitude + dt * euler_rate(attitude, new angular_velocity)
        euler_rate_x = avx_new - ca.sin(ay) * avz_new
        euler_rate_y = ca.cos(ax) * avy_new + ca.cos(ay) * ca.sin(ax) * avz_new
        euler_rate_z = -ca.sin(ax) * avy_new + ca.cos(ay
                                                      ) * ca.cos(ax) * avz_new
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
    dyn = SimpleDynamics()
    new_state = dyn.simulate_quadrotor(
        torch.tensor([action]), torch.tensor([state]), 0.05
    )
    print("new state simple", new_state)

    dyn_mpc = SimpleDynamicsMPC()
    F = dyn_mpc.drone_dynamics_simple(0.05)
    new_state_mpc = F(np.array(state), np.array(action))
    print("new state mpc", new_state_mpc)
