import numpy as np

from aria_teleop.filters import OneEuroFilter, OneEuroRotationFilter
from aria_teleop.geometry import (
    heading_frame,
    interpolate_rotation,
    invert_transform,
    make_transform,
    rot_x,
    rot_z,
    rotation_angle,
)


def test_invert_transform():
    T = make_transform(rot_z(0.7) @ rot_x(0.3), [0.1, -0.2, 0.3])
    np.testing.assert_allclose(T @ invert_transform(T), np.eye(4), atol=1e-12)


def test_interpolate_rotation_halfway():
    R = interpolate_rotation(np.eye(3), rot_z(1.0), 0.5)
    np.testing.assert_allclose(R, rot_z(0.5), atol=1e-12)


def test_heading_frame_is_right_handed_and_level():
    # Head looking forward along room +y and pitched down 30 degrees.
    forward = np.array([0.0, np.cos(np.radians(30)), -np.sin(np.radians(30))])
    A = heading_frame(forward, up=[0, 0, 1])
    np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(A), 1.0)
    np.testing.assert_allclose(A @ [0, 1, 0], [1, 0, 0], atol=1e-12)  # facing direction -> robot forward
    np.testing.assert_allclose(A @ [-1, 0, 0], [0, 1, 0], atol=1e-12)  # operator's left -> robot left


def test_heading_frame_yaw_offset_and_degenerate():
    A = heading_frame([1, 0, 0], [0, 0, 1], yaw_offset=np.pi)
    np.testing.assert_allclose(A @ [1, 0, 0], [-1, 0, 0], atol=1e-12)
    assert heading_frame([0, 0, -1], [0, 0, 1]) is None


def test_one_euro_removes_jitter_on_still_signal():
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.5, beta=10.0)
    raw = np.array([0.4, -0.1, 0.2]) + rng.normal(0, 0.003, (300, 3))
    out = np.array([f(x, i / 30) for i, x in enumerate(raw)])
    assert out[100:].std(axis=0).max() < raw[100:].std(axis=0).max() / 2


def test_one_euro_rotation_converges():
    f = OneEuroRotationFilter(min_cutoff=1.5, beta=2.0)
    f(np.eye(3), 0.0)
    for i in range(1, 120):
        R = f(rot_z(0.5), i / 30)
    assert rotation_angle(R.T @ rot_z(0.5)) < 1e-3
