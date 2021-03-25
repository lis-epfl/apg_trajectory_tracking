import numpy as np
import torch
import casadi as ca

m = 1.01
I_xx = 0.04766
rho = 1.225
S = 0.276
c = 0.185
g = 9.81
# linearized for alpha = 0 and u = 12 m/s
Cl0 = 0.3900
Cl_alpha = 4.5321
Cl_q = 0.3180
Cl_del_e = 0.527
# drag coefficients
Cd0 = 0.0765
Cd_alpha = 0.3346
Cd_q = 0.354
Cd_del_e = 0.004
# moment coefficients
Cm0 = 0.0200
Cm_alpha = -1.4037
Cm_q = -0.1324
Cm_del_e = -0.4236

# lower and upper bounds:
alpha_bound = float(5 / 180 * np.pi)

torch_pi = np.pi


def long_dynamics(state, action, dt):
    # extract variables
    # state
    x = state[:, 0]  # x position
    h = state[:, 1]  # z position / altitude
    u = state[:, 2]  # forward velocity in body frame
    w = state[:, 3]  # upward velocity  in body frame
    theta = state[:, 4]  # pitch angle
    q = state[:, 5]  # pitch rate

    # input states
    T = action[:, 0] * 7
    del_e = torch_pi * (action[:, 1] * 40 - 20) / 180

    ## aerodynamic forces calculations
    # (see beard & mclain, 2012, p. 44 ff)
    V = torch.sqrt(u**2 + w**2)  # velocity norm
    alpha = torch.arctan(w / u)  # angle of attack
    alpha = torch.clamp(alpha, -alpha_bound, alpha_bound)

    # NOTE: usually all of Cl, Cd, Cm,... depend on alpha, q, delta_e
    # lift coefficient
    Cl = Cl0 + Cl_alpha * alpha + Cl_q * c / (2 * V) * q + Cl_del_e * del_e
    # drag coefficient
    Cd = Cd0 + Cd_alpha * alpha + Cd_q * c / (2 * V) * q + Cd_del_e * del_e
    # pitch moment coefficient
    Cm = Cm0 + Cm_alpha * alpha + Cm_q * c / (2 * V) * q + Cm_del_e * del_e

    # resulting forces and moment
    L = 1 / 2 * rho * V**2 * S * Cl  # lift
    D = 1 / 2 * rho * V**2 * S * Cd  # drag
    M = 1 / 2 * rho * V**2 * S * c * Cm  # pitch moment

    ## Global displacement
    x_dot = u * torch.cos(theta) + w * torch.sin(theta)  # forward
    h_dot = u * torch.sin(theta) - w * torch.cos(theta)  # upward

    ## Body fixed accelerations
    # (see Fundamentals of airplane flight mechanics, David G. Hull, 2007)
    # Assumption: T (thrust) acts parallel to the body fixed x-axis (forward)
    # L * torch.sin(alpha) = Z, D * torch.cos(alpha) = -X
    # (projections of lift/drag into body frame)
    u_dot = -w * q + (1 / m) * (
        T + L * torch.sin(alpha) - D * torch.cos(alpha) -
        m * g * torch.sin(theta)
    )
    w_dot = u * q - (1 / m) * (
        L * torch.cos(alpha) + D * torch.sin(alpha) - m * g * torch.cos(theta)
    )

    ## Pitch acceleration
    q_dot = M / I_xx

    ## State propagation through time
    # for now: time linearized model x_{k+1} = x_k + xdot_k * dt
    state_dot = torch.vstack((x_dot, h_dot, u_dot, w_dot, q, q_dot)).t()
    # state_dot = torch.tensor([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot])
    # simple integration over time
    next_state = state + dt * state_dot
    return next_state


def fixed_wing_dynamics_mpc(self, dt):
    """
    Longitudinal dynamics for fixed wing
    """

    # --------- state vector ----------------
    px, pz, u, w, theta, q = (
        ca.SX.sym('px'), ca.SX.sym('pz'), ca.SX.sym('vx'), ca.SX.sym('vz'),
        ca.SX.sym('theta'), ca.SX.sym('q')
    )
    x_state = ca.vertcat(px, pz, u, w, theta, q)

    # -----------control command ---------------
    u_thrust, u_del_e = ca.SX.sym('thrust'), ca.SX.sym('del_e')
    u_control = ca.vertcat(u_thrust, u_del_e)

    # input states
    T = u_thrust * 7
    # angle between -20 and 20 degrees
    del_e = np.pi * (u_del_e * 40 - 20) / 180

    ## aerodynamic forces calculations
    # (see beard & mclain, 2012, p. 44 ff)
    V = ca.sqrt(u**2 + w**2)  # velocity norm
    alpha = ca.atan(w / u)  # angle of attack
    # alpha = torch.clamp(alpha, -alpha_bound, alpha_bound) TODO

    # NOTE: usually all of Cl, Cd, Cm,... depend on alpha, q, delta_e
    # lift coefficient
    Cl = Cl0 + Cl_alpha * alpha + Cl_q * c / (2 * V) * q + Cl_del_e * del_e
    # drag coefficient
    Cd = Cd0 + Cd_alpha * alpha + Cd_q * c / (2 * V) * q + Cd_del_e * del_e
    # pitch moment coefficient
    Cm = Cm0 + Cm_alpha * alpha + Cm_q * c / (2 * V) * q + Cm_del_e * del_e

    # resulting forces and moment
    L = 1 / 2 * rho * V**2 * S * Cl  # lift
    D = 1 / 2 * rho * V**2 * S * Cd  # drag
    M = 1 / 2 * rho * V**2 * S * c * Cm  # pitch moment

    ## Global displacement
    x_dot = u * ca.cos(theta) + w * ca.sin(theta)  # forward
    h_dot = u * ca.sin(theta) - w * ca.cos(theta)  # upward

    ## Body fixed accelerations
    u_dot = -w * q + (1 / m) * (
        T + L * ca.sin(alpha) - D * ca.cos(alpha) - m * g * ca.sin(theta)
    )
    w_dot = u * q - (1 / m) * (
        L * ca.cos(alpha) + D * ca.sin(alpha) - m * g * ca.cos(theta)
    )

    ## Pitch acceleration
    q_dot = M / I_xx

    x_new = px + dt * x_dot
    h_new = pz + dt * h_dot
    u_new = u + dt * u_dot
    w_new = w + dt * w_dot
    theta_new = theta + dt * q
    q_new = q + dt * q_dot

    X = ca.vertcat(x_new, h_new, u_new, w_new, theta_new, q_new)

    F = ca.Function('F', [x_state, u_control], [X], ['x', 'u'], ['ode'])
    return F
