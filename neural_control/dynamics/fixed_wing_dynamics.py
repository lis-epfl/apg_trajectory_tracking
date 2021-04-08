import numpy as np
import torch
import torch.nn as nn
import json
import os
import casadi as ca
from pathlib import Path

# lower and upper bounds:
alpha_bound = float(10 / 180 * np.pi)


class FixedWingDynamics:
    """
    Dynamics of a fixed wing drone
    """

    def __init__(self, modified_params={}):
        # Load json file with default parameters
        with open(
            os.path.join(
                Path(__file__).parent.absolute(), "config_fixed_wing.json"
            ), "r"
        ) as infile:
            self.cfg = json.load(infile)

        # pi
        self.pi = np.pi

        # update with modified parameters
        self.cfg.update(modified_params)

        self.I = torch.tensor(
            [
                [self.cfg["I_xx"], 0, -self.cfg["I_xz"]],
                [0, self.cfg["I_yy"], 0],
                [-self.cfg["I_xz"], 0, self.cfg["I_zz"]]
            ]
        )

    def normalize_action(self, thrust, ome_x, ome_y, ome_z):
        T = thrust * 7
        del_e = self.pi * (ome_x * 40 - 20) / 180
        del_a = self.pi * (ome_y * 5 - 2.5) / 180
        del_r = self.pi * (ome_z * 40 - 20) / 180
        return T, del_e, del_a, del_r

    @staticmethod
    def body_wind_function(alpha, beta):
        """
        This function calculates the rotation matrix from the wind to the body
        """
        sa = torch.sin(alpha)
        sb = torch.sin(beta)
        ca = torch.cos(alpha)
        cb = torch.cos(beta)
        zero_vec = torch.zeros(sa.size())
        # create matrix
        m1 = torch.transpose(torch.vstack((ca * cb, -ca * sb, -sa)), 0, 1)
        m2 = torch.transpose(torch.vstack((sb, cb, zero_vec)), 0, 1)
        m3 = torch.transpose(torch.vstack((sa * cb, -sa * sb, ca)), 0, 1)
        R_bw = torch.stack((m1, m2, m3), dim=1)
        return R_bw

    @staticmethod
    def inertial_body_function(phi, theta, psi):
        """
        returns rotation matrix to rotate Vector from body to inertial frame
        (ZYX convention)
        i.e. vec_inertial = R_ib * vec_body
        """
        sph = torch.sin(phi)
        cph = torch.cos(phi)
        sth = torch.sin(theta)
        cth = torch.cos(theta)
        sps = torch.sin(psi)
        cps = torch.cos(psi)

        # create matrix
        vec1 = (cth * cps, cth * sps, -sth)
        m1 = torch.transpose(torch.vstack(vec1), 0, 1)
        vec2 = (
            -cph * sps + sph * sth * cps, cph * cps + sph * sth * sps,
            sph * cth
        )
        m2 = torch.transpose(torch.vstack(vec2), 0, 1)
        vec3 = (
            sph * sps + cph * sth * cps, -sph * cps + cph * sth * sps,
            cph * cth
        )
        m3 = torch.transpose(torch.vstack(vec3), 0, 1)
        R_ib = torch.stack((m1, m2, m3), dim=1)
        return torch.transpose(R_ib, 1, 2)

    def __call__(self, state, action, dt):
        return self.simulate_fixed_wing(state, action, dt)

    def simulate_fixed_wing(self, state, action, dt):
        """
        Dynamics of a fixed wing drone
        """
        # STATE
        pos = state[:, :3]  # position in inertial frame North East Down (NED)

        vel = state[:, 3:6]  # velocity in body frame
        vel_u = state[:, 3]  # forward
        vel_v = state[:, 4]  # right
        vel_w = state[:, 5]  # down

        # eul = state[:, 6:9] #  euler angles (Tait-Bryan ZYX convention)
        eul_phi = state[:, 6]  # roll
        eul_theta = state[:, 7]  # pitch
        eul_psi = state[:, 8]  # yaw

        omega = state[:, 9:12]  # angulat velocity in body frame
        ome_p = state[:, 9]  # around x
        ome_q = state[:, 10]  # around y
        ome_r = state[:, 11]  # around z

        # CONTROL SIGNAl - must be scaled
        T, del_e, del_a, del_r = self.normalize_action(
            action[:, 0], action[:, 1], action[:, 2], action[:, 3]
        )

        # multiply gravity and mass
        g_m = self.cfg["g"] * self.cfg["mass"]

        # # aerodynamic forces calculations
        # (see beard & mclain, 2012, p. 44 ff)
        V = torch.sqrt(vel_u**2 + vel_v**2 + vel_w**2)  # velocity norm
        alpha = torch.arctan(vel_w / vel_u)  # angle of attack
        alpha = torch.clamp(alpha, -alpha_bound, alpha_bound)  # TODO
        beta = torch.arctan(vel_v / V)
        beta = torch.clamp(beta, -alpha_bound, alpha_bound)
        # TODO: clamp beta?

        # NOTE: usually all of Cl, Cd, Cm,... depend on alpha, q, delta_e
        # lift coefficient
        CL = self.cfg["CL0"] + self.cfg["CL_alpha"] * alpha + self.cfg[
            "CL_q"] * self.cfg["c"] / (2 * V
                                       ) * ome_q + self.cfg["CL_del_e"] * del_e
        # drag coefficient
        CD = self.cfg["CD0"] + self.cfg["CD_alpha"] * alpha + self.cfg[
            "CD_q"] * self.cfg["c"] / (2 * V
                                       ) * ome_q + self.cfg["CD_del_e"] * del_e
        # lateral force coefficient
        CY = self.cfg["CY0"] + self.cfg["CY_beta"] * beta + self.cfg[
            "CY_p"] * self.cfg["b"] / (2 * V) * ome_p + self.cfg[
                "CY_r"] * self.cfg["b"] / (2 * V) * ome_r + self.cfg[
                    "CY_del_a"] * del_a + self.cfg["CY_del_r"] * del_r
        # roll moment coefficient
        Cl = self.cfg["Cl0"] + self.cfg["Cl_beta"] * beta + self.cfg[
            "Cl_p"] * self.cfg["b"] / (2 * V) * ome_p + self.cfg[
                "Cl_r"] * self.cfg["b"] / (2 * V) * ome_r + self.cfg[
                    "Cl_del_a"] * del_a + self.cfg["Cl_del_r"] * del_r
        # pitch moment coefficient
        Cm = self.cfg["Cm0"] + self.cfg["Cm_alpha"] * alpha + self.cfg[
            "Cm_q"] * self.cfg["c"] / (2 * V
                                       ) * ome_q + self.cfg["Cm_del_e"] * del_e
        # yaw moment coefficient
        Cn = self.cfg["Cn0"] + self.cfg["Cn_beta"] * beta + self.cfg[
            "Cn_p"] * self.cfg["b"] / (2 * V) * ome_p + self.cfg[
                "Cn_r"] * self.cfg["b"] / (2 * V) * ome_r + self.cfg[
                    "Cn_del_a"] * del_a + self.cfg["Cn_del_r"] * del_r

        #  resulting forces and moment
        L = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * CL  # lift
        D = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * CD  # drag
        Y = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * CY  # lat. force
        l = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * self.cfg[
            "c"] * Cl  # roll moment
        m = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * self.cfg[
            "c"] * Cm  # pitch moment
        n = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * self.cfg[
            "c"] * Cn  # yaw moment

        #  resulting forces (aerodynamic, weight, and propulsive)
        #  - aerodynamic forces are generally in the wind frame and must be
        # translated into forces in the body frame by the wind-to-body rotation
        # matrix (f_body = R_wind_to_body *[-DY-L]) (Beard et al., 2012,p.18).
        #  - the weight force is given in the inertial frame and must be trans-
        #    latedinto the body fixed frame by the R_inertia_to_body matrix.
        #  thrust acts in the body fixed x-z-plane at a downward facing angle
        #  of 10 forces in the body frame
        zero_vec = torch.zeros(eul_theta.size())
        epsilon_vec = zero_vec + self.cfg["epsilon"]
        vec1 = torch.unsqueeze(torch.stack((-D, Y, -L), 1), 2)
        vec3 = torch.stack(
            (
                T * torch.cos(epsilon_vec), torch.zeros(T.size()),
                T * torch.sin(epsilon_vec)
            ), 1
        )
        body_wind_matrix = self.body_wind_function(alpha, beta)
        body_to_inertia = torch.transpose(
            self.inertial_body_function(eul_phi, eul_theta, zero_vec), 1, 2
        )
        zero = torch.zeros(1, 1)
        gravity_vec = torch.vstack([zero, zero, torch.tensor(g_m)])
        f_xyz = (
            torch.matmul(body_wind_matrix, vec1) +
            torch.matmul(body_to_inertia, gravity_vec) +
            torch.unsqueeze(vec3, 2)
        )

        # moment vector in the body frame
        moment_body_frame = torch.stack((l, m, n), 1)

        # #  Global displacemen
        # displacement of the drone in the inertial frame
        # (Beard et al., 2012, p.36)
        # position change in inertial coordinates
        pos_dot = torch.matmul(
            self.inertial_body_function(eul_phi, eul_theta, eul_psi),
            torch.unsqueeze(vel, 2)
        )

        # # Body fixed accelerations
        # see Small Unmanned Aircraft, Beard et al., 2012, p.36
        uvw_dot = (1 / self.cfg["mass"]
                   ) * f_xyz[:, :, 0] - torch.cross(omega, vel, dim=1)

        # # Change in pitch attitude (change in euler angles)
        # see Small Unmanned Aircraft, Beard et al., 2012, p.36
        vec1 = (
            torch.ones(eul_phi.size()
                       ), torch.sin(eul_phi) * torch.tan(eul_theta),
            torch.cos(eul_phi) * torch.tan(eul_theta)
        )
        m1 = torch.transpose(torch.vstack(vec1), 0, 1)
        vec2 = (
            torch.zeros(eul_theta.size()), torch.cos(eul_phi),
            -torch.sin(eul_phi)
        )
        m2 = torch.transpose(torch.vstack(vec2), 0, 1)
        vec3 = (
            torch.zeros(eul_theta.size()
                        ), torch.sin(eul_phi) / torch.cos(eul_theta),
            torch.cos(eul_phi) / torch.cos(eul_theta)
        )
        m3 = torch.transpose(torch.vstack(vec3), 0, 1)
        R_bw = torch.stack((m1, m2, m3), dim=1)

        omega_uns = torch.unsqueeze(omega, 2)
        eul_angle_dot = torch.matmul(R_bw, omega_uns)

        # #  Pitch acceleration
        # Euler's equation (rigid body dynamics)
        # Ix  = (M-cross(omega,I*omega)) --> solve for x
        cross_prod = moment_body_frame - torch.cross(
            omega, torch.matmul(self.I, omega_uns)[:, :, 0], dim=1
        )
        omega_dot = torch.matmul(
            torch.inverse(self.I), torch.unsqueeze(cross_prod, 2)
        )

        # # State propagation through time
        state_dot = torch.hstack(
            (
                pos_dot[:, :, 0], uvw_dot, eul_angle_dot[:, :,
                                                         0], omega_dot[:, :, 0]
            )
        )
        # simple integration over time
        next_state = state + dt * state_dot

        return next_state


class LearntFixedWingDynamics(torch.nn.Module, FixedWingDynamics):
    """
    Trainable dynamics for a fixed wing drone
    """

    def __init__(self, modified_params={}):
        FixedWingDynamics.__init__(self, modified_params)
        super(LearntFixedWingDynamics, self).__init__()

        # trainable parameters
        self.I = nn.Parameter(
            torch.tensor(
                [
                    [self.cfg["I_xx"], 0, -self.cfg["I_xz"]],
                    [0, self.cfg["I_yy"], 0],
                    [-self.cfg["I_xz"], 0, self.cfg["I_zz"]]
                ],
                requires_grad=True
            )
        )
        # Parameter dictionary of other parameters
        dict_pytorch = {}
        for key, val in self.cfg.items():
            if "I_" in key:
                # make inertia separately
                continue
            dict_pytorch[key] = torch.nn.Parameter(
                torch.tensor([val]), requires_grad=True
            )
        self.cfg = torch.nn.ParameterDict(dict_pytorch)

        # further layer for dynamic (non parameter) mismatch
        self.linear_state_1 = nn.Linear(16, 64)
        torch.nn.init.constant_(self.linear_state_1.weight, 0)
        torch.nn.init.constant_(self.linear_state_1.bias, 0)

        self.linear_state_2 = nn.Linear(64, 12)
        torch.nn.init.constant_(self.linear_state_2.weight, 0)
        torch.nn.init.constant_(self.linear_state_2.bias, 0)

    def state_transformer(self, state, action):
        state_action = torch.cat((state, action), dim=1)
        layer_1 = torch.relu(self.linear_state_1(state_action))
        new_state = self.linear_state_2(layer_1)
        # TODO: activation function?
        return new_state

    def forward(self, state, action, dt):
        # run through D1
        new_state = self.simulate_fixed_wing(state, action, dt)
        # run through T
        added_new_state = self.state_transformer(state, action)
        return new_state + added_new_state


class FixedWingDynamicsMPC(FixedWingDynamics):

    def __init__(self, modified_params={}):
        super().__init__(modified_params)

        self.derive_parameters()

    def derive_parameters(self):
        # gravity vector
        self.cfg["g_m"] = self.cfg["g"] * self.cfg["mass"]
        self.gravity_vec_ca = ca.SX(np.array([0, 0, self.cfg["g_m"]]))
        # inertia
        self.I = torch.tensor(
            [
                [self.cfg["I_xx"], 0, -self.cfg["I_xz"]],
                [0, self.cfg["I_yy"], 0],
                [-self.cfg["I_xz"], 0, self.cfg["I_zz"]]
            ]
        )
        self.I_inv = torch.inverse(self.I)
        self.I_casadi = ca.SX(self.I.numpy())  # TODO
        self.I_inv_casadi = ca.SX(self.I_inv.numpy())

    def simulate_fixed_wing(self, dt):
        """
        Longitudinal dynamics for fixed wing
        """

        # --------- state vector ----------------
        (
            px, py, pz, vel_u, vel_v, vel_w, eul_phi, eul_theta, eul_psi,
            ome_p, ome_q, ome_r
        ) = (
            ca.SX.sym('px'), ca.SX.sym('py'), ca.SX.sym('pz'),
            ca.SX.sym('vel_u'), ca.SX.sym('vel_v'), ca.SX.sym('vel_w'),
            ca.SX.sym('eul_phi'), ca.SX.sym('eul_theta'), ca.SX.sym('eul_psi'),
            ca.SX.sym('ome_p'), ca.SX.sym('ome_q'), ca.SX.sym('ome_r')
        )
        x_state = ca.vertcat(
            px, py, pz, vel_u, vel_v, vel_w, eul_phi, eul_theta, eul_psi,
            ome_p, ome_q, ome_r
        )

        # -----------control command ---------------
        u_thrust, u_del_e, u_del_a, u_del_r = (
            ca.SX.sym('thrust'), ca.SX.sym('del_e'), ca.SX.sym('del_a'),
            ca.SX.sym('del_r')
        )
        u_control = ca.vertcat(u_thrust, u_del_e, u_del_a, u_del_r)

        # normalize action
        T, del_e, del_a, del_r = self.normalize_action(
            u_thrust, u_del_e, u_del_a, u_del_r
        )

        ## aerodynamic forces calculations
        # (see beard & mclain, 2012, p. 44 ff)
        V = ca.sqrt(vel_u**2 + vel_v**2 + vel_w**2)  # velocity norm
        alpha = ca.atan(vel_w / vel_u)  # angle of attack
        # alpha = torch.clamp(alpha, -alpha_bound, alpha_bound) TODO
        beta = ca.atan(vel_v / V)

        # NOTE: usually all of Cl, Cd, Cm,... depend on alpha, q, delta_e
        # lift coefficient
        CL = self.cfg["CL0"] + self.cfg["CL_alpha"] * alpha + self.cfg[
            "CL_q"] * self.cfg["c"] / (2 * V
                                       ) * ome_q + self.cfg["CL_del_e"] * del_e
        # drag coefficient
        CD = self.cfg["CD0"] + self.cfg["CD_alpha"] * alpha + self.cfg[
            "CD_q"] * self.cfg["c"] / (2 * V
                                       ) * ome_q + self.cfg["CD_del_e"] * del_e
        # lateral force coefficient
        CY = self.cfg["CY0"] + self.cfg["CY_beta"] * beta + self.cfg[
            "CY_p"] * self.cfg["b"] / (2 * V) * ome_p + self.cfg[
                "CY_r"] * self.cfg["b"] / (2 * V) * ome_r + self.cfg[
                    "CY_del_a"] * del_a + self.cfg["CY_del_r"] * del_r
        # roll moment coefficient
        Cl = self.cfg["Cl0"] + self.cfg["Cl_beta"] * beta + self.cfg[
            "Cl_p"] * self.cfg["b"] / (2 * V) * ome_p + self.cfg[
                "Cl_r"] * self.cfg["b"] / (2 * V) * ome_r + self.cfg[
                    "Cl_del_a"] * del_a + self.cfg["Cl_del_r"] * del_r
        # pitch moment coefficient
        Cm = self.cfg["Cm0"] + self.cfg["Cm_alpha"] * alpha + self.cfg[
            "Cm_q"] * self.cfg["c"] / (2 * V
                                       ) * ome_q + self.cfg["Cm_del_e"] * del_e
        # yaw moment coefficient
        Cn = self.cfg["Cn0"] + self.cfg["Cn_beta"] * beta + self.cfg[
            "Cn_p"] * self.cfg["b"] / (2 * V) * ome_p + self.cfg[
                "Cn_r"] * self.cfg["b"] / (2 * V) * ome_r + self.cfg[
                    "Cn_del_a"] * del_a + self.cfg["Cn_del_r"] * del_r

        # resulting forces and moment
        L = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * CL  # lift
        D = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * CD  # drag
        Y = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * CY  # lateral force
        l = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * self.cfg[
            "c"] * Cl  # roll moment
        m = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * self.cfg[
            "c"] * Cm  # pitch moment
        n = 1 / 2 * self.cfg["rho"] * V**2 * self.cfg["S"] * self.cfg[
            "c"] * Cn  # yaw moment

        vec3 = ca.vertcat(
            T * ca.cos(self.cfg["epsilon"]), 0,
            T * ca.sin(self.cfg["epsilon"])
        )
        # body wind function
        sa = ca.sin(alpha)
        sb = ca.sin(beta)
        ca_ = ca.cos(alpha)
        cb = ca.cos(beta)
        vec1_multiplied = ca.vertcat(
            ca_ * cb * (-D) + (-ca_) * sb * Y - sa * (-L),
            sb * (-D) + cb * Y,
            sa * cb * (-D) - sa * sb * Y + ca_ * (-L)
        )
        # inertia body function
        sph = ca.sin(eul_phi)
        cph = ca.cos(eul_phi)
        sth = ca.sin(eul_theta)
        cth = ca.cos(eul_theta)
        sps = ca.sin(eul_psi)
        cps = ca.cos(eul_psi)
        vec2 = ca.vertcat(
            -self.cfg["g_m"] * sth, sph * cth * self.cfg["g_m"],
            cph * cth * self.cfg["g_m"]
        )
        f_xyz = vec1_multiplied + vec2 + vec3

        moment_body_frame = ca.vertcat(l, m, n)

        # position dot
        px_dot = (
            vel_u * (cth * cps) + vel_v * (-cph * sps + sph * sth * cps) +
            vel_w * (sph * sps + cph * sth * cps)
        )
        py_dot = vel_u * cth * sps + vel_v * (
            cph * cps + sph * sth * sps
        ) + vel_w * (-sph * cps + cph * sth * sps)
        pz_dot = -vel_u * sth + sph * cth * vel_v + vel_w * cth * cph

        # uvw_dot
        omega = ca.vertcat(ome_p, ome_q, ome_r)
        vel = ca.vertcat(vel_u, vel_v, vel_w)
        uvw_dot = (1 / self.cfg["mass"]) * f_xyz - ca.cross(omega, vel)
        # pos[3] und uvw[3] ist falsch

        # change in attitude
        tth = ca.tan(eul_theta)
        eul_phi_dot = ome_p + sph * tth * ome_q + cph * tth * ome_r
        eul_theta_dot = cph * ome_q - sph * ome_r
        eul_psi_dot = sph / cth * ome_q + cph / cth * ome_r

        # Pitch acceleration
        cross_prod = moment_body_frame - ca.cross(omega, self.I_casadi @ omega)
        omega_dot = self.I_inv_casadi @ cross_prod

        x_dot = ca.vertcat(
            px_dot, py_dot, pz_dot, uvw_dot, eul_phi_dot, eul_theta_dot,
            eul_psi_dot, omega_dot[0], omega_dot[1], omega_dot[2]
        )

        X = x_state + dt * x_dot

        F = ca.Function('F', [x_state, u_control], [X], ['x', 'u'], ['ode'])
        return F


if __name__ == "__main__":
    state_test_np = np.array(
        [
            0.6933, -0.8747, 0.9757, -0.8422, 0.5494, -1.1936, 0.0368, 0.8417,
            -0.9412, -1.4291, 0.4538, -0.5257
        ]
    )
    state_test = torch.unsqueeze(torch.from_numpy(state_test_np), 0).float()
    action_test_np = np.array([-0.5518, -2.9553, 0.0311, -0.6691])
    action_test = torch.unsqueeze(torch.from_numpy(action_test_np), 0).float()
    print(state_test[0])
    print(action_test[0])
    normal_dyn = FixedWingDynamics()
    next_state = normal_dyn.simulate_fixed_wing(state_test, action_test, 0.05)
    print("------------")
    print(next_state[0])

    # test: compare to mpc
    # if test doesnt work, remove clamp!!
    mpc_dyn = FixedWingDynamicsMPC()
    F = mpc_dyn.simulate_fixed_wing(0.05)
    mpc_state = F(state_test_np, action_test_np)
    print("--------------------")
    print(mpc_state)
