import numpy as np
import pytest

from aria_teleop.fusion import HandFrame, HandFuser, HandState, PoseBuffer
from aria_teleop.geometry import make_transform, rot_z, rotation_angle
from aria_teleop.mapping import PoseSmoother, Status, TeleopConfig, TeleopMapper
from aria_teleop.sources import _R_ROOM_DEVICE_LEVEL, MockSource

HOME = np.array([0.3, 0.0, 0.2])
DT = 1 / 30
WRIST0 = np.array([0.4, -0.15, -0.4])


def head(yaw=0.0):
    return make_transform(rot_z(yaw) @ _R_ROOM_DEVICE_LEVEL, [0, 0, 0])


def frame(t, wrist, R=np.eye(3), head_T=None, pinch=0.05, conf=0.9, other_pinch=None, session=0):
    hands = {}
    if wrist is not None:
        lm = np.tile(wrist, (21, 1))
        lm[1] = lm[0] + [0.0, pinch, 0.0]
        hands["right"] = HandState(make_transform(R, wrist), lm, conf)
    if other_pinch is not None:
        lm = np.tile([0.35, 0.2, -0.4], (21, 1))
        lm[1] = lm[0] + [0.0, other_pinch, 0.0]
        hands["left"] = HandState(make_transform(np.eye(3), lm[0]), lm, conf)
    return HandFrame(
        t_ns=int(round(t * 1e9)),
        received_at=t,
        hands=hands,
        T_room_head=head() if head_T is None else head_T,
        gravity_room=np.array([0.0, 0.0, -9.81]),
        session=session,
    )


def feed(mapper, t0, n, **kw):
    """Feed n frames at 30 Hz starting at t0; returns (last command, next t)."""
    t = t0
    for _ in range(n):
        cmd = mapper.update(frame(t, **kw), t)
        t += DT
    return cmd, t


def engaged_mapper(**cfg):
    m = TeleopMapper(TeleopConfig(**cfg), HOME, np.eye(3))
    m.set_clutch(True)
    return m


def test_engaging_does_not_move_the_robot():
    m = engaged_mapper()
    cmd, _ = feed(m, 0.0, 10, wrist=WRIST0)
    assert cmd.status is Status.TRACKING
    np.testing.assert_allclose(cmd.position, HOME, atol=1e-9)


def test_hand_translation_maps_to_robot_frame():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 5, wrist=WRIST0)
    cmd, _ = feed(m, t, 60, wrist=WRIST0 + [0.10, 0.05, -0.03])  # forward, left, down
    np.testing.assert_allclose(cmd.position, HOME + [0.10, 0.05, -0.03], atol=1e-3)


def test_heading_follows_where_the_operator_faces():
    # Operator faces room +y: moving the hand along +y is "forward" for the robot.
    m = engaged_mapper()
    facing_y = head(np.pi / 2)
    _, t = feed(m, 0.0, 5, wrist=WRIST0, head_T=facing_y)
    cmd, _ = feed(m, t, 60, wrist=WRIST0 + [0.0, 0.10, 0.0], head_T=facing_y)
    np.testing.assert_allclose(cmd.position, HOME + [0.10, 0.0, 0.0], atol=1e-3)


def test_head_turning_after_engage_does_not_change_the_mapping():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 5, wrist=WRIST0)
    cmd, _ = feed(m, t, 60, wrist=WRIST0 + [0.10, 0.0, 0.0], head_T=head(0.6))
    np.testing.assert_allclose(cmd.position, HOME + [0.10, 0.0, 0.0], atol=1e-3)


def test_heading_persists_across_reclutch_until_reset():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 5, wrist=WRIST0)  # first engage: facing room +x
    m.set_clutch(False)
    _, t = feed(m, t, 3, wrist=WRIST0)
    m.set_clutch(True)
    facing_y = head(np.pi / 2)
    _, t = feed(m, t, 3, wrist=WRIST0, head_T=facing_y)  # re-engage with head turned
    cmd, t = feed(m, t, 60, wrist=WRIST0 + [0.10, 0.0, 0.0], head_T=facing_y)
    np.testing.assert_allclose(cmd.position, HOME + [0.10, 0.0, 0.0], atol=1e-3)  # still room +x = forward

    m.reset_heading()
    start = cmd.position.copy()
    _, t = feed(m, t, 3, wrist=WRIST0 + [0.10, 0.0, 0.0], head_T=facing_y)
    cmd, _ = feed(m, t, 60, wrist=WRIST0 + [0.10, 0.10, 0.0], head_T=facing_y)
    np.testing.assert_allclose(cmd.position, start + [0.10, 0.0, 0.0], atol=1e-3)  # now room +y = forward


def test_scale():
    m = engaged_mapper(scale=0.5)
    _, t = feed(m, 0.0, 5, wrist=WRIST0)
    cmd, _ = feed(m, t, 60, wrist=WRIST0 + [0.10, 0.0, 0.0])
    np.testing.assert_allclose(cmd.position, HOME + [0.05, 0.0, 0.0], atol=1e-3)


def test_wrist_rotation_maps_to_gripper_rotation():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 5, wrist=WRIST0)
    cmd, _ = feed(m, t, 90, wrist=WRIST0, R=rot_z(0.3))
    assert rotation_angle(cmd.rotation.T @ rot_z(0.3)) < 2e-3


def test_no_rotation_keeps_orientation():
    m = engaged_mapper(use_rotation=False)
    _, t = feed(m, 0.0, 5, wrist=WRIST0)
    cmd, _ = feed(m, t, 30, wrist=WRIST0, R=rot_z(0.3))
    np.testing.assert_allclose(cmd.rotation, np.eye(3))


def test_single_frame_glitch_is_rejected():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 10, wrist=WRIST0)
    cmd = m.update(frame(t, WRIST0 + [0.3, 0.0, 0.0]), t)
    assert m.rejected_frames == 1
    np.testing.assert_allclose(cmd.position, HOME, atol=1e-9)
    cmd, _ = feed(m, t + DT, 5, wrist=WRIST0)
    np.testing.assert_allclose(cmd.position, HOME, atol=1e-6)


def test_lost_hand_holds_then_reanchors_without_jump():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 10, wrist=WRIST0)
    cmd, t = feed(m, t, 15, wrist=None)  # 0.5 s with no hand
    assert cmd.status is Status.HOLD
    held = cmd.position.copy()
    # Hand comes back 20 cm away: robot must not jump there.
    cmd, t = feed(m, t, 5, wrist=WRIST0 + [0.2, 0.0, 0.0])
    assert cmd.status is Status.TRACKING
    np.testing.assert_allclose(cmd.position, held, atol=1e-6)
    cmd, _ = feed(m, t, 60, wrist=WRIST0 + [0.25, 0.0, 0.0])
    np.testing.assert_allclose(cmd.position, held + [0.05, 0.0, 0.0], atol=1e-3)


def test_stale_stream_holds():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 10, wrist=WRIST0)
    last = frame(t - DT, WRIST0)
    cmd = m.update(last, t + 1.0)  # same old frame, a second later
    assert cmd.status is Status.HOLD


def test_released_clutch_ignores_the_hand():
    m = TeleopMapper(TeleopConfig(), HOME, np.eye(3))
    cmd, _ = feed(m, 0.0, 30, wrist=WRIST0 + [0.2, 0.0, 0.0])
    assert cmd.status is Status.IDLE
    np.testing.assert_allclose(cmd.position, HOME)


def test_target_stays_in_workspace():
    m = engaged_mapper()
    _, t = feed(m, 0.0, 5, wrist=WRIST0)
    for k in range(1, 40):  # walk 2 m forward slowly so nothing is rejected as a glitch
        cmd, t = feed(m, t, 1, wrist=WRIST0 + [0.05 * k, 0.0, 0.0])
    cfg = TeleopConfig()
    assert np.linalg.norm(cmd.position - np.array(cfg.shoulder)) <= cfg.max_reach_m + 1e-9
    assert cmd.position[0] <= cfg.workspace_max[0] + 1e-9


@pytest.mark.parametrize("pinch, expected", [(0.02, 0.0), (0.05, 0.5), (0.08, 1.0)])
def test_pinch_maps_to_gripper(pinch, expected):
    m = engaged_mapper()
    cmd, _ = feed(m, 0.0, 60, wrist=WRIST0, pinch=pinch)
    assert cmd.gripper == pytest.approx(expected, abs=0.02)


def test_pinch_clutch_is_a_deadman():
    m = TeleopMapper(TeleopConfig(clutch="pinch"), HOME, np.eye(3))
    cmd, t = feed(m, 0.0, 5, wrist=WRIST0, other_pinch=0.06)
    assert cmd.status is Status.IDLE
    cmd, t = feed(m, t, 5, wrist=WRIST0, other_pinch=0.01)
    assert cmd.status is Status.TRACKING
    cmd, t = feed(m, t, 5, wrist=WRIST0, other_pinch=0.035)  # inside hysteresis band: stays on
    assert cmd.status is Status.TRACKING
    cmd, t = feed(m, t, 10, wrist=WRIST0)  # clutch hand leaves view
    assert cmd.status is Status.IDLE


def test_pose_buffer():
    buf = PoseBuffer(max_extrapolation_ns=50_000_000)
    buf.add(0, make_transform(np.eye(3), [0, 0, 0]))
    buf.add(100_000_000, make_transform(rot_z(0.2), [1, 0, 0]))
    T = buf.lookup(50_000_000)
    np.testing.assert_allclose(T[:3, 3], [0.5, 0, 0])
    assert rotation_angle(T[:3, :3].T @ rot_z(0.1)) < 1e-9
    assert buf.lookup(140_000_000) is not None  # short extrapolation holds the last pose
    assert buf.lookup(200_000_000) is None
    assert buf.lookup(-1) is None
    buf.add(300_000_000, np.eye(4), session="new")  # new VIO session drops old poses
    assert buf.lookup(50_000_000) is None


def test_fuser_cancels_head_motion():
    clock = [0.0]
    fuser = HandFuser(clock=lambda: clock[0])
    mock = MockSource(fuser, noise_m=0.0, dropout=0.0, glitch=0.0, head_yaw_deg=30.0)
    for i in range(400):  # 2 s of head poses at 200 Hz, hands every 7th sample
        t = i / 200
        clock[0] = t
        fuser.on_head_pose(int(round(t * 1e9)), mock.head_pose(t), session=0)
        if i % 7 == 0:
            fuser.on_hands(int(round(t * 1e9)), mock.observe(t))
            truth = mock.wrist_pose(t)
            got = fuser.latest().hands["right"].T_room_wrist
            np.testing.assert_allclose(got[:3, 3], truth[:3, 3], atol=1e-9)
            assert rotation_angle(got[:3, :3].T @ truth[:3, :3]) < 1e-9


def test_smoother_spreads_steps_and_caps_speed():
    s = PoseSmoother(blend_time=0.035, max_speed=0.6)
    s.reset(np.zeros(3), np.eye(3))
    p, _ = s.step(np.array([0.01, 0, 0]), np.eye(3), 0.002)
    assert 0 < p[0] < 0.01  # not a step
    for _ in range(20):
        p, _ = s.step(np.array([0.01, 0, 0]), np.eye(3), 0.002)
    assert p[0] == pytest.approx(0.01)
    s.reset(np.zeros(3), np.eye(3))
    p, _ = s.step(np.array([1.0, 0, 0]), np.eye(3), 0.002)
    assert p[0] == pytest.approx(0.6 * 0.002)
