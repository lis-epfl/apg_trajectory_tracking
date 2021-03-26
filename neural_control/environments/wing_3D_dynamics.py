import numpy as np
import torch

mass = 1.01
I_xx = 0.04766  #  moment of inertia around x
I_yy = 0.05005  #  moment of inertia around y
I_zz = 0.09558  #  moment of inertia around z
I_xz = -0.00105  #  moment of inertia around xz

# moment of inertia tensor
I = torch.tensor([[I_xx, 0, -I_xz], [0, I_yy, 0], [-I_xz, 0, I_zz]])
I_inv = torch.inverse(I)

rho = 1.225  # air pressure [kg/m^3]
S = 0.276  # wing surface
c = 0.185  # chord length
b = 1.54  # wing span
g = 9.81
# linearized for alpha = 0 and u = 12 m/s
CL0 = 0.3900
CL_alpha = 4.5321
CL_q = 0.3180
CL_del_e = 0.527
# drag coefficients
CD0 = 0.0765
CD_alpha = 0.3346
CD_q = 0.354
CD_del_e = 0.004
CY0 = 0
CY_beta = -0.033
CY_p = -0.100
CY_r = 0.039
CY_del_a = 0.000
CY_del_r = 0.225
Cl0 = 0
Cl_beta = -0.081
Cl_p = -0.529
Cl_r = 0.159
Cl_del_a = -0.453
Cl_del_r = 0.005
# moment coefficients
Cm0 = 0.0200
Cm_alpha = -1.4037
Cm_q = -0.1324
Cm_del_e = -0.4236
Cn0 = 0
Cn_beta = 0.189
Cn_p = -0.083
Cn_r = -0.948
Cn_del_a = -0.041
Cn_del_r = -0.077
epsilon = 1 / 18 * np.pi

# predefine vector
gravity_vec = torch.unsqueeze(torch.tensor([0, 0, mass * g]), 1)

# lower and upper bounds:
alpha_bound = float(5 / 180 * np.pi)
torch_pi = np.pi


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
        -cph * sps + sph * sth * cps, cph * cps + sph * sth * sps, sph * cth
    )
    m2 = torch.transpose(torch.vstack(vec2), 0, 1)
    vec3 = (
        sph * sps + cph * sth * cps, -sph * cps + cph * sth * sps, cph * cth
    )
    m3 = torch.transpose(torch.vstack(vec3), 0, 1)
    R_ib = torch.stack((m1, m2, m3), dim=1)
    return torch.transpose(R_ib, 1, 2)


def long_dynamics(state, action, dt):
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
    T = action[:, 0] * 7
    del_e = torch_pi * (action[:, 1] * 40 - 20) / 180
    del_a = torch_pi * (action[:, 2] * 5 - 2.5) / 180
    del_r = torch_pi * (action[:, 3] * 40 - 20) / 180

    # # aerodynamic forces calculations
    # (see beard & mclain, 2012, p. 44 ff)
    V = torch.sqrt(vel_u**2 + vel_v**2 + vel_w**2)  # velocity norm
    alpha = torch.arctan(vel_w / vel_u)  # angle of attack
    # alpha = torch.clamp(alpha, -alpha_bound, alpha_bound) # TODO
    beta = torch.arctan(vel_v / V)
    # TODO: clamp beta?

    # NOTE: usually all of Cl, Cd, Cm,... depend on alpha, q, delta_e
    # lift coefficient
    CL = CL0 + CL_alpha * alpha + CL_q * c / (2 * V) * ome_q + CL_del_e * del_e
    # drag coefficient
    CD = CD0 + CD_alpha * alpha + CD_q * c / (2 * V) * ome_q + CD_del_e * del_e
    # lateral force coefficient
    CY = CY0 + CY_beta * beta + CY_p * b / (2 * V) * ome_p + CY_r * b / (
        2 * V
    ) * ome_r + CY_del_a * del_a + CY_del_r * del_r
    # roll moment coefficient
    Cl = Cl0 + Cl_beta * beta + Cl_p * b / (2 * V) * ome_p + Cl_r * b / (
        2 * V
    ) * ome_r + Cl_del_a * del_a + Cl_del_r * del_r
    # pitch moment coefficient
    Cm = Cm0 + Cm_alpha * alpha + Cm_q * c / (2 * V) * ome_q + Cm_del_e * del_e
    # yaw moment coefficient
    Cn = Cn0 + Cn_beta * beta + Cn_p * b / (2 * V) * ome_p + Cn_r * b / (
        2 * V
    ) * ome_r + Cn_del_a * del_a + Cn_del_r * del_r

    #  resulting forces and moment
    L = 1 / 2 * rho * V**2 * S * CL  # lift
    D = 1 / 2 * rho * V**2 * S * CD  # drag
    Y = 1 / 2 * rho * V**2 * S * CY  # lateral force
    l = 1 / 2 * rho * V**2 * S * c * Cl  # roll moment
    m = 1 / 2 * rho * V**2 * S * c * Cm  # pitch moment
    n = 1 / 2 * rho * V**2 * S * c * Cn  # yaw moment

    #  resulting forces (aerodynamic, weight, and propulsive)
    #  - aerodynamic forces are generally in the wind frame and must be
    #    translated into forces in the body frame by the wind-to-body rotation
    #    matrix (f_body = R_wind_to_body *[-DY-L]) (Beard et al., 2012,p.18).
    #  - the weight force is given in the inertial frame and must be translated
    #    into the body fixed frame by the R_inertia_to_body matrix.
    #  thrust acts in the body fixed x-z-plane at a downward facing angle of 10°
    #  forces in the body frame09
    zero_vec = torch.zeros(eul_theta.size())
    epsilon_vec = zero_vec + epsilon
    vec1 = torch.unsqueeze(torch.stack((-D, Y, -L), 1), 2)
    vec3 = torch.stack(
        (
            T * torch.cos(epsilon_vec), torch.zeros(T.size()),
            T * torch.sin(epsilon_vec)
        ), 1
    )
    body_wind_matrix = body_wind_function(alpha, beta)
    body_to_inertia = torch.transpose(
        inertial_body_function(eul_phi, eul_theta, zero_vec), 1, 2
    )
    f_xyz = (
        torch.matmul(body_wind_matrix, vec1) +
        torch.matmul(body_to_inertia, gravity_vec) + torch.unsqueeze(vec3, 2)
    )

    # moment vector in the body frame
    moment_body_frame = torch.stack((l, m, n), 1)

    # #  Global displacemen
    # displacement of the drone in the inertial frame
    # (Beard et al., 2012, p.36)
    # position change in inertial coordinates
    pos_dot = torch.matmul(
        inertial_body_function(eul_phi, eul_theta, eul_psi),
        torch.unsqueeze(vel, 2)
    )

    # # Body fixed accelerations
    # see Small Unmanned Aircraft, Beard et al., 2012, p.36
    uvw_dot = (1 / mass) * f_xyz[:, :, 0] - torch.cross(omega, vel, dim=1)

    # # Change in pitch attitude (change in euler angles)
    # see Small Unmanned Aircraft, Beard et al., 2012, p.36
    vec1 = (
        torch.ones(eul_phi.size()), torch.sin(eul_phi) * torch.tan(eul_theta),
        torch.cos(eul_phi) * torch.tan(eul_theta)
    )
    m1 = torch.transpose(torch.vstack(vec1), 0, 1)
    vec2 = (
        torch.zeros(eul_theta.size()), torch.cos(eul_phi), -torch.sin(eul_phi)
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
        omega, torch.matmul(I, omega_uns)[:, :, 0], dim=1
    )
    omega_dot = torch.matmul(I_inv, torch.unsqueeze(cross_prod, 2))

    # # State propagation through time
    state_dot = torch.hstack(
        (
            pos_dot[:, :, 0], uvw_dot, eul_angle_dot[:, :, 0], omega_dot[:, :,
                                                                         0]
        )
    )
    # simple integration over time
    next_state = state + dt * state_dot

    return next_state


if __name__ == "__main__":
    state_test = torch.tensor(
        [
            [
                0.6933, -0.8747, 0.9757, -0.8422, 0.5494, -1.1936, 0.0368,
                0.8417, -0.9412, -1.4291, 0.4538, -0.5257
            ],
            [
                0.0947, -0.4377, -0.5699, -0.8684, -0.7914, -0.2026, -0.8172,
                1.6205, 0.7867, -0.3465, -0.0789, -2.1874
            ]
        ]
    )
    action_test = torch.tensor(
        [
            [-0.5518, -2.9553, 0.0311, -0.6691],
            [-1.1911, -0.6097, 0.8386, 1.6545]
        ]
    )
    print(state_test[0])
    print(action_test[0])
    next_state = long_dynamics(state_test, action_test, 0.05)
    print("------------")
    print(next_state[0])
