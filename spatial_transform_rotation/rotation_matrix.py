import numpy as np


def _deg2rad(deg):
    return deg/180.0*np.pi


def _rz(roll):
    rad = _deg2rad(roll)
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]])
    return rot_mat


def _ry(yaw):
    rad = _deg2rad(yaw)
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]])
    return rot_mat


def _rx(pitch):
    rad = _deg2rad(pitch)
    c = np.cos(rad)
    s = np.sin(rad)
    rot_mat = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]])
    return rot_mat


def scale_matrix(s):
    s_mat = np.array([
        [s, 0, 0],
        [0, s, 0],
        [0, 0, s]])
    return s_mat


def rotation_matrix(roll=.0, pitch=.0, yaw=.0, order='rpy'):
    """
    :param roll: Amount to rotate on Z axis in degrees
    :param pitch: Amount to rotate on X axis in degrees
    :param yaw: Amount to rotate on Y (up-)axis in degrees
    :param order: Order of rotations is Roll > Pitch > Yaw by default
    :return: R = Yaw * Pitch * Roll

    Convention used in this function: Y is up axis and Z is depth axis.
    This rotation matrix can be applied to a 3D tensor [B,C,D,H,W] of shape where
    the corresponding axis for the shape are [-,-,Z,X,Y]
    """
    rz = _rz(roll)
    rx = _rx(pitch)
    ry = _ry(yaw)
    if order == 'rpy':
        r = np.matmul(ry, np.matmul(rx, rz))
    elif order == 'ypr':
        r = np.matmul(rz, np.matmul(rx, ry))
    else:
        raise NotImplementedError()
    return r
