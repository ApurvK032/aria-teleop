"""Joint pipeline: IK, shaper guarantees, the simulated arm, and the Trossen backend against a fake driver."""

import sys
import types

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("trossen_arm_mujoco")

from scipy.spatial.transform import Rotation  # noqa: E402

from aria_teleop.arm import (  # noqa: E402
    REST_Q, ArmController, ArmFault, ArmKinematics, ArmIK, JointShaper, SimArm, TrossenArm, log_rotation,
)
from aria_teleop.geometry import rot_y, rot_z  # noqa: E402
from aria_teleop.mapping import PoseSmoother, Status, TeleopCommand  # noqa: E402
from aria_teleop.sim import HOME_POSITION, HOME_ROTATION, WidowXSim  # noqa: E402

DT = 0.005


def test_log_rotation_matches_scipy():
    for R in Rotation.random(500, random_state=2).as_matrix():
        np.testing.assert_allclose(log_rotation(R), Rotation.from_matrix(R).as_rotvec(), atol=1e-5)
    np.testing.assert_allclose(log_rotation(np.eye(3)), 0.0)


def test_ik_reaches_poses_and_stays_inside_the_ranges():
    sim = WidowXSim(PoseSmoother())
    kin = ArmKinematics(sim.model)
    ik = ArmIK(kin)
    q_home, err_p, err_r = ik.solve(HOME_POSITION, HOME_ROTATION, REST_Q, iterations=300)
    assert err_p < 1e-4 and err_r < 1e-3
    q, err_p, err_r = ik.solve(HOME_POSITION + [0.05, 0.05, 0.05], rot_z(0.3) @ rot_y(0.2), q_home, iterations=100)
    assert err_p < 1e-3 and err_r < 1e-2
    assert np.all(q >= ik.lower) and np.all(q <= ik.upper)


def run_shaper(target, n=800, **kw):
    s = JointShaper(np.zeros(1), [1.5], kw.pop("a", [4.0]), kw.pop("j", [240.0]), lower=[-3.0], upper=[kw.pop("upper", 3.0)])
    out = np.array([s.step(np.array([target(i * DT)]), DT) for i in range(n)])[:, :, 0]
    return out[:, 0], out[:, 1], out[:, 2]


@pytest.mark.parametrize("a, j", [(2.0, 10.0), (4.0, 60.0), (4.0, 240.0), (16.0, 960.0)])
def test_shaper_steps_without_overshoot_and_respects_its_caps(a, j):
    q, v, acc = run_shaper(lambda t: 0.0 if t < 0.05 else 2.0, n=1200, a=[a], j=[j])
    assert q.max() <= 2.0 + 1e-6  # a 2 rad jump (say, a glitch) is closed without overshoot
    assert np.abs(v).max() <= 1.5 + 1e-9
    assert np.abs(acc).max() <= a + 1e-9
    assert (np.abs(np.diff(acc)) / DT).max() <= j * 1.001
    assert abs(q[-1] - 2.0) < 0.01


def test_shaper_never_passes_a_range_end():
    q, _, _ = run_shaper(lambda t: 1.2, n=400, upper=0.2)
    assert q.max() <= 0.2 + 1e-12


def test_shaper_tracks_smooth_motion_closely():
    target = lambda t: 0.3 * np.sin(np.pi * t)  # 0.3 rad at 0.5 Hz
    q, _, _ = run_shaper(target, n=1200)
    t = np.arange(1200) * DT
    assert np.mean(np.abs(q - target(t))[400:]) < 0.015


def preview_controller(**kw):
    sim = WidowXSim(PoseSmoother())
    arm = ArmController(SimArm(sim), ArmKinematics(sim.model), PoseSmoother(), (HOME_POSITION, HOME_ROTATION), sim=sim, **kw)
    arm.start(0.0)
    return sim, arm


def test_preview_moves_home_then_follows_the_target_smoothly():
    sim, arm = preview_controller()
    p_home = arm.kin.fk(sim.data.qpos[sim._qpos].copy())[0]
    assert np.linalg.norm(p_home - HOME_POSITION) < 0.01  # the start move got there
    t, speeds = 0.0, []
    for i in range(400):  # 2 s sliding 6 cm sideways
        t = i * DT
        cmd = TeleopCommand(HOME_POSITION + [0.0, 0.03 * min(t, 2.0), 0.0], HOME_ROTATION, 1.0, Status.TRACKING)
        arm.advance(cmd, t + DT)
        speeds.append(arm.joint_speeds())
    for _ in range(200):
        t += DT
        arm.advance(cmd, t + DT)
    assert arm.tracking_error() < 0.005
    assert np.max(speeds) < 1.0


def test_gate_rejects_a_command_outside_the_joint_ranges():
    _, arm = preview_controller()
    q = arm.q_cmd.copy()
    q[2] = arm.gate_lower[2] - 0.01
    with pytest.raises(ArmFault):
        arm._gate(q, DT)


# --- Trossen backend against a fake driver ---------------------------------------


class FakeDriver:
    calls: list = []
    firmware = "1.8.3"

    def __init__(self):
        self.q = np.zeros(7)

    def configure(self, model, end_effector, ip, clear_error, timeout=20.0):
        self.calls.append(("configure", ip))

    def get_controller_version(self):
        return self.firmware

    def get_joint_limits(self):
        lim = types.SimpleNamespace
        ranges = [(-3.05, 3.05), (0.0, 3.14), (0.0, 2.36), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14), (0.0, 0.044)]
        return [lim(position_min=lo, position_max=hi, velocity_max=3.0) for lo, hi in ranges]

    def get_arm_positions(self):
        return list(self.q[:6])

    def set_arm_modes(self, mode):
        self.calls.append(("set_arm_modes", mode))

    def set_gripper_mode(self, mode):
        self.calls.append(("set_gripper_mode", mode))

    def set_arm_positions(self, q, goal_time=2.0, blocking=True, ff_v=None, ff_a=None):
        assert len(q) == 6 and (ff_v is None or len(ff_v) == 6)
        self.calls.append(("set_arm_positions", round(goal_time, 3), blocking))
        if blocking:
            self.q[:6] = q  # the move completes
        else:
            self.q[:6] = q  # the arm keeps up perfectly in this fake

    def set_gripper_position(self, g, goal_time=2.0, blocking=True, ff_v=0.0, ff_a=0.0):
        self.calls.append(("set_gripper_position", round(goal_time, 3), blocking))

    def cleanup(self, reboot_controller=False):
        self.calls.append(("cleanup",))


@pytest.fixture
def fake_trossen(monkeypatch):
    module = types.SimpleNamespace(
        TrossenArmDriver=FakeDriver,
        Model=types.SimpleNamespace(wxai_v0="wxai_v0"),
        StandardEndEffector=types.SimpleNamespace(wxai_v0_follower="follower"),
        Mode=types.SimpleNamespace(position="position", idle="idle"),
        __version__="1.8.6",
    )
    monkeypatch.setitem(sys.modules, "trossen_arm", module)
    monkeypatch.setattr("socket.create_connection", lambda *a, **k: types.SimpleNamespace(close=lambda: None))
    FakeDriver.calls = []
    FakeDriver.firmware = "1.8.3"
    return FakeDriver


def trossen_controller(read_only):
    sim = WidowXSim(PoseSmoother())
    backend = TrossenArm("192.168.1.2", read_only=read_only)
    return ArmController(backend, ArmKinematics(sim.model), PoseSmoother(), (HOME_POSITION, HOME_ROTATION), sim=sim)


def drive(arm, seconds=0.5):
    for i in range(int(seconds / DT)):
        cmd = TeleopCommand(HOME_POSITION + [0.0, 0.02, 0.0], HOME_ROTATION, 1.0, Status.TRACKING)
        arm.advance(cmd, (i + 1) * DT)


def test_dry_run_never_moves_the_arm(fake_trossen):
    arm = trossen_controller(read_only=True)
    info = arm.start(0.0)
    assert "firmware 1.8.3" in info and "driver 1.8.6" in info
    drive(arm)
    arm.stop()
    names = [c[0] for c in fake_trossen.calls]
    assert names == ["configure", "cleanup"]  # no mode change, no position command


def test_real_arm_lifecycle(fake_trossen):
    arm = trossen_controller(read_only=False)
    arm.start(0.0)
    drive(arm)
    arm.stop()
    calls = fake_trossen.calls
    names = [c[0] for c in calls]
    assert names[0] == "configure" and names[-1] == "cleanup"
    blocking_moves = [c for c in calls if c[0] == "set_arm_positions" and c[2]]
    assert len(blocking_moves) == 2  # rest -> home at start, home -> rest at stop
    streamed = [c for c in calls if c[0] == "set_arm_positions" and not c[2]]
    assert len(streamed) > 50 and all(c[1] == 0.02 for c in streamed[:-1])


def test_unreachable_arm_fails_fast(fake_trossen, monkeypatch):
    def refuse(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr("socket.create_connection", refuse)
    arm = trossen_controller(read_only=True)
    with pytest.raises(ArmFault, match="can't reach"):
        arm.start(0.0)
    assert fake_trossen.calls == []  # the driver was never asked to connect


def test_firmware_mismatch_is_refused(fake_trossen):
    fake_trossen.firmware = "1.11.0"
    arm = trossen_controller(read_only=False)
    with pytest.raises(ArmFault, match="doesn't match"):
        arm.start(0.0)
    assert not any(c[0] in ("set_arm_modes", "set_arm_positions") for c in fake_trossen.calls)
    assert fake_trossen.calls[-1][0] == "cleanup"


def test_tracking_error_fault_stops_and_returns_to_rest(fake_trossen):
    arm = trossen_controller(read_only=False)
    arm.start(0.0)
    arm.backend._driver.set_arm_positions = lambda *a, **k: None  # the arm stops following its commands
    real_read = arm.backend.read
    arm.backend.read = lambda: real_read() + 0.2
    with pytest.raises(ArmFault, match="from its command"):
        drive(arm)
    del arm.backend.read
    arm.backend._driver.set_arm_positions = FakeDriver.set_arm_positions.__get__(arm.backend._driver)
    arm.stop()
    assert fake_trossen.calls[-1][0] == "cleanup"
    assert any(c[0] == "set_arm_positions" and c[2] for c in fake_trossen.calls[-4:])  # moved to rest
