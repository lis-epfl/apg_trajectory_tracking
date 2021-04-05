import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2

# target state means that theta is zero --> only third position matters
target_state = 0  # torch.from_numpy(np.array([0, 0, 0, 0]))

# DEFINE VARIABLES
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = (masspole + masscart)
length = 0.5  # actually half the pole's length
polemass_length = (masspole * length)
max_force_mag = 40.0
tau = 0.02  # seconds between state updates
muc = 0.0005
mup = 0.000002


def simulate_cartpole(state, action):
    """
    Compute new state from state and action
    """
    # get state
    x = state[:, 0]
    x_dot = state[:, 1]
    theta = state[:, 2]
    theta_dot = state[:, 3]
    # (x, x_dot, theta, theta_dot) = state

    # helper variables
    force = max_force_mag * action
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)
    sig = muc * torch.sign(x_dot)

    # add and multiply
    temp = torch.add(
        torch.squeeze(force),
        polemass_length * torch.mul(theta_dot**2, sintheta)
    )
    # divide
    thetaacc = (
        gravity * sintheta - (costheta * (temp - sig)) -
        (mup * theta_dot / polemass_length)
    ) / (length * (4.0 / 3.0 - masspole * costheta * costheta / total_mass))

    # swapped these two lines
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    # add velocity of cart
    xacc = (temp - (polemass_length * thetaacc * costheta) - sig) / total_mass
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc

    new_state = torch.stack((x, x_dot, theta, theta_dot), dim=1)
    return new_state
