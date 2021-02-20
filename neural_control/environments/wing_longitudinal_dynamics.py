import numpy as np
import torch

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
    T = action[:, 0]  # thrust
    del_e = action[:, 1]  # angle of elevator

    ## aerodynamic forces calculations
    # (see beard & mclain, 2012, p. 44 ff)
    V = torch.sqrt(u**2 + w**2)  # velocity norm
    alpha = torch.arctan(w / u)  # angle of attack

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
    h_dot = u * torch.sin(theta) + w * torch.cos(theta)  # upward

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
        L * torch.cos(alpha) - D * torch.sin(alpha) - m * g * torch.cos(theta)
    )

    ## Change in pitch attitude
    theta_dot = q

    ## Pitch acceleration
    q_dot = M / I_xx

    ## State propagation through time
    # for now: time linearized model x_{k+1} = x_k + xdot_k * dt
    state_dot = torch.vstack((x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot)
                             ).t()
    # state_dot = torch.tensor([x_dot, h_dot, u_dot, w_dot, theta_dot, q_dot])
    # simple integration over time
    next_state = state + dt * state_dot
    return next_state


# # constants
# dummy = constants[0]
# m        = constants[1] # mass
# I_xx     = constants[2] # moment of inertia around x
# rho      = constants[3] # air pressure [kg/m**3]
# S        = constants[4] # wing surface
# c        = constants[5] # chord length
# g        = constants[6] # gravity
# # aerodynamic parameters vehicles (see beard & mclain, 2012, p. 44 ff)
# # lift coefficients
# Cl0      = constants[7]
# Cl_alpha = constants[8]
# Cl_q     = constants[9]
# Cl_del_e = constants[10]
# # drag coefficients
# Cd0      = constants[11]
# Cd_alpha = constants[12]
# Cd_q     = constants[13]
# Cd_del_e = constants[14]
# # moment coefficients
# Cm0      = constants[15]
# Cm_alpha = constants[16]
# Cm_q     = constants[17]
# Cm_del_e = constants[18]