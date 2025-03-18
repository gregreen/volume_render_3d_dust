import numpy as np

import tensorflow as tf
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d, quaternion


def interpolate_rotation_matrices(R0, R1, t):
    """
    Interpolates between two 3D rotation matrices using SLERP (spherical linear interpolation) via quaternions.

    Args:
        R0 (tf.Tensor): A 3x3 rotation matrix representing the initial orientation.
        R1 (tf.Tensor): A 3x3 rotation matrix representing the final orientation.
        t (float or tf.Tensor): Interpolation factor in the range [0, 1].

    Returns:
        tf.Tensor: A 3x3 rotation matrix representing the interpolated rotation.
    """
    q0 = quaternion.from_rotation_matrix(R0)
    q1 = quaternion.from_rotation_matrix(R1)
    dot = tf.reduce_sum(q0 * q1)
    q1 = q1 * tf.sign(dot)
    dot = tf.reduce_sum(q0 * q1)
    theta = tf.acos(tf.clip_by_value(dot, -1.0, 1.0))
    sin_theta = tf.sin(theta)
    q = (tf.sin((1 - t) * theta) / sin_theta) * q0 + (tf.sin(t * theta) / sin_theta) * q1
    return rotation_matrix_3d.from_quaternion(q)


def get_rotation_matrix(lon, lat, roll):
    """
    Computes a rotation matrix based on given Euler angles (longitude, latitude, and roll).

    Args:
        lon (float): Longitude angle in degrees.
        lat (float): Latitude angle in degrees.
        roll (float): Roll angle in degrees.

    Returns:
        tf.Tensor: A 3x3 rotation matrix computed from the input angles.
    """
    R2 = rotation_matrix_3d.from_axis_angle(
        tf.cast([0., 0., 1.], tf.float32),
        tf.cast([np.radians(lon - 90)], tf.float32)
    )
    R1 = rotation_matrix_3d.from_axis_angle(
        tf.cast([1., 0., 0.], tf.float32),
        tf.cast([np.radians(lat - 90)], tf.float32)
    )
    R0 = rotation_matrix_3d.from_axis_angle(
        tf.cast([0., 0., 1.], tf.float32),
        tf.cast([-np.radians(roll)], tf.float32)
    )
    R = tf.einsum('ij,jk,kl->il', R2, R1, R0)
    return R