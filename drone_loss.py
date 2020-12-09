import torch
from environments.copter import copter_params


class AttitudeLoss():

    def __init__(
        self,
        angle_factor=1.0,
        angvel_factor=1e-2,
        attitude_error_transform=None,
        angvel_error_transform=None
    ):
        self._angle_factor = angle_factor
        self._angvel_factor = angvel_factor
        self._angle_err_trafo = attitude_error_transform
        self._angvel_err_trafo = angvel_error_transform

    def calculate_loss(self, state):
        # TODO: extract pitch, role and yaw and angular velocity
        attitude = state.attitude
        angle_error = self.angle_error(attitude)
        avel_error = self.velocity_error(state.angular_velocity)

        if self._angle_err_trafo:
            angle_error = self._angle_err_trafo(angle_error)

        if self._angvel_err_trafo:
            avel_error = self._angvel_err_trafo(avel_error)

        # compute final loss
        loss = self._angle_factor * angle_error
        loss += self._angvel_factor * avel_error

        return loss

    def angle_error(self, attitude):
        return attitude.pitch**2 + attitude.roll**2 + attitude.yaw**2

    def velocity_error(self, angular_velocity):
        # TODO: axis
        return torch.sum(angular_velocity**2)

    def update_parameters(
        self,
        angle_factor=None,
        angvel_factor=None,
        angle_error_transform=None,
        angvel_error_transform=None
    ):
        if angle_factor is not None:
            self._angle_factor = angle_factor

        if angvel_factor is not None:
            self._angvel_factor = angvel_factor

        if angle_error_transform is not None:
            self._angle_err_trafo = angle_error_transform

        if angvel_error_transform is not None:
            self._angvel_err_trafo = angvel_error_transform

    def __str__(self):
        return "AttitudeReward(%g, %g)" % (
            self._angle_factor, self._angvel_factor
        )


def control_loss(current_state, action):
    """
    Computes loss for applying an action to the current state by comparing to
    the target state
    Arguments:
        current_state: array with x entries describing attitude and velocity
        action: control signal of dimension 4 (thrust of rotors)
    """

    pass
