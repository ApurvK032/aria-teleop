"""Small SE(3) / SO(3) helpers shared by the fusion and mapping stages.

Conventions: a 4x4 transform ``T_a_b`` maps points from frame b into frame a.
Rotations are 3x3 matrices; quaternions only appear at the MuJoCo boundary, in
MuJoCo's [w, x, y, z] order.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = np.asarray(translation).reshape(3)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return make_transform(R.T, -R.T @ T[:3, 3])


def rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def from_rotvec(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(v).as_matrix()


def interpolate_rotation(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Slerp from R0 (alpha=0) to R1 (alpha=1)."""
    return R0 @ from_rotvec(alpha * rotvec(R0.T @ R1))


def interpolate_transform(T0: np.ndarray, T1: np.ndarray, alpha: float) -> np.ndarray:
    return make_transform(
        interpolate_rotation(T0[:3, :3], T1[:3, :3], alpha),
        (1.0 - alpha) * T0[:3, 3] + alpha * T1[:3, 3],
    )


def rotation_angle(R: np.ndarray) -> float:
    return float(np.linalg.norm(rotvec(R)))


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def heading_frame(
    forward: np.ndarray, up: np.ndarray, yaw_offset: float = 0.0
) -> np.ndarray | None:
    """Rotation taking room-frame vectors into a gravity-aligned operator frame.

    Rows are (forward, left, up), matching the robot base convention
    (x forward, y left, z up), so ``A @ v_room`` reads "how far forward / left /
    up did the hand move from where the operator is facing". ``yaw_offset``
    (radians, positive = turn left) corrects for an operator who does not face
    the same way as the robot. Returns None when ``forward`` is within ~12 degrees
    of vertical, where heading is undefined.
    """
    up = np.asarray(up, dtype=float)
    up = up / np.linalg.norm(up)
    f = np.asarray(forward, dtype=float)
    f = f - np.dot(f, up) * up
    n = np.linalg.norm(f)
    if n < 0.2:
        return None
    f = f / n
    if yaw_offset:
        f = f * np.cos(yaw_offset) + np.cross(up, f) * np.sin(yaw_offset)
    left = np.cross(up, f)
    return np.vstack([f, left, up])


def quat_wxyz(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_quat(scalar_first=True)
