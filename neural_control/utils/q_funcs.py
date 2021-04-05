import numpy as np
import pyquaternion
import casadi as cs


def project_to_line(a, b, p):
    """
    Project a n-dim position p onto a line spanned by a and b
    """
    # define points a and b on the line and p as the current position
    if np.all(a == b):
        return a
    ap = p - a
    ab = np.expand_dims(b - a, 1)
    dot = np.dot(ab, ab.T)
    norm = np.sum(ab**2)
    result = a + np.dot(dot, ap) / norm
    return result


def euler_to_quaternion(roll, pitch, yaw):
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
        roll / 2
    ) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.array([qw, qx, qy, qz])


def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q**2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy)
                ],
                [
                    2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx)
                ],
                [
                    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2)
                ]
            ]
        )

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(
                1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy)
            ),
            cs.horzcat(
                2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx)
            ),
            cs.horzcat(
                2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2)
            )
        )

    return rot_mat


def q_dot_new(q, w):
    t0 = -0.5 * (w[0] * q[1] + w[1] * q[2] + w[2] * q[3])
    t1 = 0.5 * (w[0] * q[0] + w[1] * q[3] - w[2] * q[2])
    t2 = 0.5 * (w[1] * q[0] + w[2] * q[1] - w[0] * q[3])
    t3 = 0.5 * (w[2] * q[0] + w[0] * q[2] - w[1] * q[1])
    return np.array([t0, t1, t2, t3])


def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words,
     rotates quaternion q by r. Quaternion format:
    wxyz.
    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the
     input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return cs.vertcat(t0, t1, t2, t3)


def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix.
    :param rot: 3x3 numpy array, representing a valid rotation matrix
    :return: a quaternion corresponding to the 3D rotation described by the
     input matrix. Quaternion format: wxyz
    """

    q = pyquaternion.Quaternion(matrix=rot)
    return np.array([q.w, q.x, q.y, q.z])


def undo_quaternion_flip(q_past, q_current):
    """
    Detects if q_current generated a quaternion jump and corrects it. Requires
     knowledge of the previous quaternion
    in the series, q_past
    :param q_past: 4-dimensional vector representing a quaternion in wxyz form.
    :param q_current: 4-dimensional vector representing a quaternion in wxyz
     form. Will be corrected if it generates
    a flip wrt q_past.
    :return: q_current with the flip removed if necessary
    """

    if np.sqrt(np.sum((q_past - q_current)**2)
               ) > np.sqrt(np.sum((q_past + q_current)**2)):
        return -q_current
    return q_current


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)
    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data
     type as v
    """

    if isinstance(v, np.ndarray):
        return np.array(
            [
                [0, -v[0], -v[1], -v[2]], [v[0], 0, v[2], -v[1]],
                [v[1], -v[2], 0, v[0]], [v[2], v[1], -v[0], 0]
            ]
        )

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]), cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]), cs.horzcat(v[2], v[1], -v[0], 0)
    )


def decompose_quaternion(q):
    """
    Decomposes a quaternion into a z rotation and an xy rotation
    :param q: 4-dimensional numpy array of CasADi MX (format qw, qx, qy, qz)
    :return: two 4-dimensional arrays (same format as input), where the first
     contains the xy rotation and the second
    the z rotation, in quaternion forms.
    """

    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, cs.MX):
        qz = unit_quat(cs.vertcat(w, 0, 0, z))
    else:
        qz = unit_quat(np.array([w, 0, 0, z]))
    qxy = q_dot_q(q, quaternion_inverse(qz))

    return qxy, qz


def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)