"""End-to-end in MuJoCo: mock glasses -> fusion -> mapper -> WidowX AI, on simulated time."""

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("trossen_arm_mujoco")

from aria_teleop.fusion import HandFuser  # noqa: E402
from aria_teleop.geometry import heading_frame, rot_y  # noqa: E402
from aria_teleop.mapping import PoseSmoother, Status, TeleopCommand, TeleopConfig, TeleopMapper  # noqa: E402
from aria_teleop.sim import HOME_POSITION, HOME_ROTATION, WidowXSim, run  # noqa: E402
from aria_teleop.sources import MockSource  # noqa: E402


def drive(seconds, engage_at=1.0, **mock_kw):
    clock = [0.0]
    fuser = HandFuser(clock=lambda: clock[0])
    mock = MockSource(fuser, **mock_kw)
    sim = WidowXSim(PoseSmoother())
    mapper = TeleopMapper(TeleopConfig(), HOME_POSITION, HOME_ROTATION)
    dt = sim.timestep
    steps_per_head, steps_per_hand = round(1 / 200 / dt), round(1 / 30 / dt)
    log = []
    for i in range(int(seconds / dt)):
        t = i * dt
        clock[0] = t
        if i % steps_per_head == 0:
            fuser.on_head_pose(int(round(t * 1e9)), mock.head_pose(t), session=0)
        if i % steps_per_hand == 0:
            fuser.on_hands(int(round(t * 1e9)), mock.observe(t))
        if t >= engage_at:
            mapper.set_clutch(True)
        cmd = mapper.update(fuser.latest(), t)
        sim.step(cmd)
        log.append((t, cmd, sim.ee_pose()[0], sim.tracking_error()))
    return mock, mapper, log


def test_arm_follows_scripted_hand_and_ignores_head_sway():
    mock, mapper, log = drive(9.0, noise_m=0.0, dropout=0.0, glitch=0.0, head_yaw_deg=25.0)
    tracking = [entry for entry in log if entry[1].status is Status.TRACKING]
    assert len(tracking) > 0.8 * len(log) * (8 / 9)
    # The robot target should reproduce the hand path relative to the engage moment,
    # rotated into the heading the head had at engage, with later head sway removed.
    t_anchor = min(t for t, *_ in tracking)
    A = heading_frame(mock.head_pose(t_anchor)[:3, 2], [0.0, 0.0, 1.0])
    w0 = mock.wrist_pose(t_anchor)[:3, 3]
    p0 = tracking[0][1].position
    errs = []
    for t, cmd, ee, _ in tracking[200:]:  # skip the first 0.4 s while the filter settles
        expected = p0 + A @ (mock.wrist_pose(t)[:3, 3] - w0)
        errs.append(np.linalg.norm(cmd.position - expected))
    assert np.mean(errs) < 0.01  # filter lag + 30 Hz sample-and-hold at ~0.1 m/s
    # And the simulated arm actually gets there.
    assert np.mean([e for *_, e in tracking]) < 0.01
    assert all(np.all(np.isfinite(ee)) for *_, ee, _ in log)


def test_noisy_stream_stays_stable():
    _, mapper, log = drive(8.0, noise_m=0.003, dropout=0.05, glitch=0.02, seed=1)
    assert mapper.rejected_frames > 0
    positions = np.array([cmd.position for _, cmd, _, _ in log])
    step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert step.max() < 0.05  # no glitch made it through as a jump
    tracking = [e for _, cmd, _, e in log if cmd.status is Status.TRACKING]
    assert np.mean(tracking) < 0.015


def peak_joint_speed_for_a_sudden_wrist_bend(cap):
    sim = WidowXSim(PoseSmoother(max_angular_speed=10.0), joint_speed_limit=0.5)
    home = TeleopCommand(HOME_POSITION, HOME_ROTATION, 0.5, Status.TRACKING)
    for _ in range(1000):  # walk to home uncapped
        sim.step(home)
    sim.cap_joint_speed = cap
    bend = TeleopCommand(HOME_POSITION, rot_y(-0.5) @ HOME_ROTATION, 0.5, Status.TRACKING)
    peak = np.zeros(6)
    for _ in range(500):
        sim.step(bend)
        peak = np.maximum(peak, sim.joint_speeds())
    return peak.max()


def test_joint_speed_cap_holds_joints_to_the_limit():
    assert peak_joint_speed_for_a_sudden_wrist_bend(cap=False) > 2.0  # what the IK asks for
    assert peak_joint_speed_for_a_sudden_wrist_bend(cap=True) <= 0.5


def test_joint_margins_measure_distance_to_the_range_ends():
    sim = WidowXSim(PoseSmoother())
    q = sim.data.qpos[sim._qpos]
    expected = np.minimum(q - sim.joint_range[:, 0], sim.joint_range[:, 1] - q)
    np.testing.assert_allclose(sim.joint_margins(), expected)
    assert sim.joint_margins()[1] == pytest.approx(0.0)  # starts folded, shoulder at the end of its range


def test_run_warns_when_a_joint_is_too_fast(capsys):
    fuser = HandFuser()
    source = MockSource(fuser)
    sim = WidowXSim(PoseSmoother(), joint_speed_limit=0.01)
    mapper = TeleopMapper(TeleopConfig(), HOME_POSITION, HOME_ROTATION)
    source.start()
    try:
        stats = run(sim, mapper, fuser, headless=True, duration=2.5, auto_engage_after=0.2)
    finally:
        source.stop()
    out = capsys.readouterr().out
    assert "WARNING joints:" in out and "rad/s >" in out
    assert stats["speed_warnings"] >= 1 and stats["joint_speed_peak"].max() > 0.01
