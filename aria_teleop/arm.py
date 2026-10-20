"""Joint-level control shared by the simulated and the real WidowX AI.

One fixed-rate loop turns the mapper's end-effector target into joint commands:

    target pose -> PoseSmoother -> ArmIK -> JointShaper -> safety gate -> ArmBackend

The loop runs at its own rate (200 Hz by default), not at the rate hand frames
arrive, so input timing never reaches the motors. ``JointShaper`` limits every
joint's velocity, acceleration and jerk and brakes early enough to stop on the
target, and the backend receives position, velocity and acceleration together
(the Trossen driver uses the last two as feedforward). The same loop drives the
MuJoCo arm (``SimArm``) and the real arm (``TrossenArm``), so the simulator shows
what the hardware will be sent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import mujoco
import numpy as np

from .geometry import quat_wxyz  # noqa: F401  (re-exported for backends)
from .mapping import PoseSmoother, TeleopCommand

ARM_JOINTS = tuple(f"joint_{i}" for i in range(6))
REST_Q = np.zeros(6)  # the folded pose the arm powers up in (the model's "home" keyframe)
GRIPPER_OPEN_M = 0.040  # gripper opening commanded for openness 1.0


class ArmFault(RuntimeError):
    """Stop teleoperation: hold the arm, then return it to rest."""


def log_rotation(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (fast enough to call every IK iteration)."""
    cos = min(1.0, max(-1.0, (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0))
    angle = math.acos(cos)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if angle > math.pi - 1e-4:  # near a half turn the antisymmetric part vanishes
        k = int(np.argmax(np.diag(R)))
        axis = R[:, k] + np.eye(3)[:, k]
        return axis / np.linalg.norm(axis) * angle
    return w * (angle / (2.0 * math.sin(angle)))


class ArmKinematics:
    """Forward kinematics and Jacobians of the arm, on its own MjData (no dynamics)."""

    def __init__(self, model: mujoco.MjModel, ee_site: str = "ee_site"):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joints])
        self.joint_range = model.jnt_range[joints].copy()  # (6, 2), rad
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qpos_adr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def jacobian(self) -> np.ndarray:
        """6x6 [linear; angular] Jacobian at the configuration of the last ``fk`` call."""
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self._jp, self._jr, self.site)
        return np.vstack([self._jp[:, self.dof_adr], self._jr[:, self.dof_adr]])


class ArmIK:
    """Damped least squares on the end-effector pose, a few iterations per control tick.

    ``rotation_weight`` (metres per radian) sets how much an orientation error counts
    against a position error when both can't be met, e.g. near joint limits. Damping
    grows near singularities, and joints stay ``limit_margin`` inside their ranges.
    """

    def __init__(
        self,
        kinematics: ArmKinematics,
        *,
        iterations: int = 3,
        rotation_weight: float = 0.5,
        damping: float = 0.01,
        singular_damping: float = 0.08,
        singular_threshold: float = 0.02,
        max_step: float = 0.25,
        limit_margin: float = 0.03,
    ):
        self.kin = kinematics
        self.iterations = iterations
        self.weights = np.array([1.0, 1.0, 1.0, rotation_weight, rotation_weight, rotation_weight])
        self.damping = damping
        self.singular_damping = singular_damping
        self.singular_threshold = singular_threshold
        self.max_step = max_step
        self.lower = kinematics.joint_range[:, 0] + limit_margin
        self.upper = kinematics.joint_range[:, 1] - limit_margin

    def solve(self, p: np.ndarray, R: np.ndarray, q_seed: np.ndarray, iterations: int | None = None):
        """Returns (q, position residual m, orientation residual rad)."""
        q = np.clip(np.asarray(q_seed, dtype=float), self.lower, self.upper)
        for _ in range(self.iterations if iterations is None else iterations):
            p_now, R_now = self.kin.fk(q)
            error = self.weights * np.concatenate([p - p_now, log_rotation(R @ R_now.T)])
            J = self.weights[:, None] * self.kin.jacobian()
            sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
            ramp = max(0.0, 1.0 - sigma_min / self.singular_threshold)
            lam2 = self.damping**2 + (self.singular_damping * ramp) ** 2
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), error)
            q = np.clip(q + np.clip(dq, -self.max_step, self.max_step), self.lower, self.upper)
        p_now, R_now = self.kin.fk(q)
        return q, float(np.linalg.norm(p - p_now)), float(np.linalg.norm(log_rotation(R @ R_now.T)))


class JointShaper:
    """Turns a jumpy joint target into a smooth trajectory, one control tick at a time.

    Each joint follows the target like a critically damped spring that also matches
    the target's velocity: smooth motion is tracked without lag, and a jump (a
    tracking glitch, a re-anchor, a limit) is closed without overshoot. Acceleration,
    jerk and velocity are capped, and a joint is never faster than it can brake from,
    jerk cap included, before the target or the end of its range.
    """

    def __init__(
        self,
        q0: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        *,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        bandwidth: float = 15.0,  # rad/s: how stiffly the target is followed
        target_velocity_smoothing: float = 0.3,
    ):
        self.v_max = np.asarray(max_velocity, dtype=float).copy()
        self.a_max = np.asarray(max_acceleration, dtype=float).copy()
        self.j_max = np.asarray(max_jerk, dtype=float).copy()
        n = len(self.v_max)
        self.lower = np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float).copy()
        self.upper = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float).copy()
        # The jerk cap delays acceleration changes by up to a_max / j_max; a stiffer
        # spring than that delay allows would overshoot and ring.
        self.bandwidth = float(np.min(np.minimum(bandwidth, 0.7 * self.j_max / self.a_max)))
        self.alpha = target_velocity_smoothing
        self.reset(q0)

    def _toward_bound(self, room: np.ndarray) -> np.ndarray:
        """Largest acceleration toward a speed bound ``room`` away (negative: above it)."""
        return np.sign(room) * np.minimum(self.a_max, np.sqrt(2.0 * self.j_max * np.abs(room)))

    def brake_speed(self, distance: np.ndarray) -> np.ndarray:
        """Fastest speed from which a joint can still stop within ``distance``.

        Stopping from speed v with the acceleration ramped at the jerk cap takes
        about v**2 / (2 a) + v a / (2 j); this solves that for v, with 20% margin.
        """
        a, b = self.a_max, self.a_max / (2.0 * self.j_max)
        d = np.maximum(np.nan_to_num(distance, posinf=1e6), 0.0)
        return 0.8 * (np.sqrt((a * b) ** 2 + 2.0 * a * d) - a * b)

    def reset(self, q: np.ndarray) -> None:
        self.q = np.asarray(q, dtype=float).copy()
        self.v = np.zeros_like(self.q)
        self.a = np.zeros_like(self.q)
        self._target: np.ndarray | None = None
        self._target_v = np.zeros_like(self.q)

    def step(self, q_target: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_target = np.asarray(q_target, dtype=float)
        if self._target is not None:
            raw = np.clip((q_target - self._target) / dt, -self.v_max, self.v_max)
            self._target_v += self.alpha * (raw - self._target_v)
        self._target = q_target.copy()
        e = np.clip(q_target, self.lower, self.upper) - self.q
        w = self.bandwidth
        a = w * w * e + 2.0 * w * (self._target_v - self.v)
        # Speed bounds: the cap, braking before either range end, and braking before
        # the target (plus however fast the target itself is moving away).
        toward = self.brake_speed(np.abs(e)) + np.abs(self._target_v)
        v_up = np.minimum(self.v_max, self.brake_speed(self.upper - self.q))
        v_down = np.minimum(self.v_max, self.brake_speed(self.q - self.lower))
        v_up = np.where(e > 0, np.minimum(v_up, toward), v_up)
        v_down = np.where(e < 0, np.minimum(v_down, toward), v_down)
        # Approach a speed bound with an acceleration the jerk cap can still ramp to zero
        # by the time the bound is reached, and actively slow down when above one;
        # then cap the jerk itself.
        a = np.minimum(a, self._toward_bound(v_up - self.v))
        a = np.maximum(a, -self._toward_bound(self.v + v_down))
        self.a = np.clip(a, self.a - self.j_max * dt, self.a + self.j_max * dt)
        v_end = np.clip(self.v + self.a * dt, -self.v_max, self.v_max)
        q = self.q + 0.5 * (self.v + v_end) * dt  # exact for constant acceleration: ticks join up smoothly
        self.v = v_end
        stopped = (q < self.lower) | (q > self.upper)  # backstop: stop dead at the range end
        self.q = np.clip(q, self.lower, self.upper)
        self.v[stopped] = 0.0
        self.a[stopped] = 0.0
        return self.q.copy(), self.v.copy(), self.a.copy()


@dataclass
class ArmState:
    q: np.ndarray  # measured arm joints, rad
    joint_range: np.ndarray  # (6, 2) rad
    velocity_max: np.ndarray | None  # rad/s per joint, as the arm reports it
    description: str


class SimArm:
    """The MuJoCo arm as an ArmBackend.

    Behaves like a controller given a goal and a goal time: every physics step, the
    servo target moves along a cubic from the current setpoint (position and
    velocity) to the latest goal, arriving at the goal time. Holding each goal until
    the next tick instead would kick the stiff servos every tick.
    """

    name = "sim"
    read_only = False

    def __init__(self, sim):
        self.sim = sim  # aria_teleop.sim.WidowXSim
        q = sim.data.qpos[sim._qpos].copy()
        self._segment = (q, np.zeros(6), q, np.zeros(6), 1.0)  # (q0, v0, q1, v1, duration)
        self._tau = 0.0

    def connect(self) -> ArmState:
        self._segment = (self.read(), np.zeros(6), self.read(), np.zeros(6), 1.0)
        return ArmState(self.read(), self.sim.joint_range.copy(), None, "MuJoCo")

    def _setpoint(self) -> tuple[np.ndarray, np.ndarray]:
        q0, v0, q1, v1, T = self._segment
        s = self._tau / T
        if s >= 1.0:
            return q1 + v1 * (self._tau - T), v1  # past the goal time: continue at the goal velocity
        h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
        d00, d10, d01, d11 = 6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s
        return h00 * q0 + h10 * T * v0 + h01 * q1 + h11 * T * v1, (d00 * q0 + d01 * q1) / T + d10 * v0 + d11 * v1

    def step(self) -> None:
        """One physics step, with the servo target taken from the current segment."""
        self._tau += self.sim.timestep
        self.sim.robot.set_arm_joint_positions(self._setpoint()[0])
        mujoco.mj_step(self.sim.model, self.sim.data)

    def move_blocking(self, q: np.ndarray, gripper_m: float, duration: float) -> None:
        """Like the driver's blocking move: a smooth (cosine) move to ``q`` over ``duration``."""
        q0 = self.read()
        self.sim.robot.set_gripper_position(gripper_m)
        steps = max(1, int(duration / self.sim.timestep))
        for i in range(1, steps + 50):
            s = 0.5 - 0.5 * math.cos(math.pi * min(i / steps, 1.0))
            self.sim.robot.set_arm_joint_positions(q0 + s * (q - q0))
            mujoco.mj_step(self.sim.model, self.sim.data)
        q = np.asarray(q, dtype=float).copy()
        self._segment = (q, np.zeros(6), q, np.zeros(6), 1.0)
        self._tau = 0.0

    def start_streaming(self) -> None:
        pass

    def send(self, q, v, a, gripper_m, goal_time=0.02) -> None:
        q0, v0 = self._setpoint()
        self._segment = (q0, v0, np.asarray(q, dtype=float).copy(), np.asarray(v, dtype=float).copy(),
                         max(float(goal_time), self.sim.timestep))
        self._tau = 0.0
        self.sim.robot.set_gripper_position(gripper_m)

    def read(self) -> np.ndarray:
        return self.sim.data.qpos[self.sim._qpos].copy()

    def hold(self) -> None:
        self.send(self.read(), np.zeros(6), np.zeros(6), self.sim.robot.get_gripper_position(), 0.1)

    def close(self) -> None:
        pass


class TrossenArm:
    """The real WidowX AI through Trossen's ``trossen_arm`` driver.

    With ``read_only`` (dry run) it connects and reads state and limits but never
    switches the joints to position mode or sends a command.
    """

    def __init__(self, robot_ip: str, *, read_only: bool = True):
        self.robot_ip = robot_ip
        self.read_only = read_only
        self.name = "dry-run" if read_only else "real"
        self._ta = None
        self._driver = None
        self._streaming = False

    CONTROLLER_PORT = 50001  # the arm controller's TCP server (the driver logs it on connect)

    def connect(self) -> ArmState:
        import socket

        import trossen_arm as ta  # the driver's major.minor must match the arm's firmware

        try:  # the driver itself can wait a long time on an unreachable address
            socket.create_connection((self.robot_ip, self.CONTROLLER_PORT), timeout=3.0).close()
        except OSError as e:
            raise ArmFault(f"can't reach the arm controller at {self.robot_ip}:{self.CONTROLLER_PORT} ({e}); "
                           "check the Ethernet cable, the arm's power, and this PC's IP on that network") from e
        self._ta = ta
        driver = ta.TrossenArmDriver()
        driver.configure(ta.Model.wxai_v0, ta.StandardEndEffector.wxai_v0_follower, self.robot_ip, False)
        self._driver = driver
        firmware = str(driver.get_controller_version())
        package = getattr(ta, "__version__", None) or _package_version("trossen-arm")
        if _major_minor(firmware) != _major_minor(package):
            self.close()
            raise ArmFault(f"trossen-arm {package} doesn't match the arm's firmware {firmware}; "
                           f"install trossen-arm=={'.'.join(map(str, _major_minor(firmware)))}.*")
        limits = driver.get_joint_limits()[:6]
        q = np.asarray(driver.get_arm_positions(), dtype=float)
        joint_range = np.array([[lim.position_min, lim.position_max] for lim in limits])
        velocity_max = np.array([lim.velocity_max for lim in limits])
        if not (np.all(np.isfinite(q)) and np.all(q >= joint_range[:, 0] - 0.05) and np.all(q <= joint_range[:, 1] + 0.05)):
            self.close()
            raise ArmFault(f"arm reports joints outside their ranges: {np.round(q, 3)}")
        return ArmState(q, joint_range, velocity_max, f"WidowX AI at {self.robot_ip}, firmware {firmware}, driver {package}")

    def _require_motion(self) -> None:
        if self.read_only:
            raise RuntimeError("dry run: motion commands are disabled")

    def move_blocking(self, q: np.ndarray, gripper_m: float, duration: float) -> None:
        self._require_motion()
        ta, d = self._ta, self._driver
        d.set_arm_modes(ta.Mode.position)
        d.set_gripper_mode(ta.Mode.position)
        d.set_gripper_position(float(gripper_m), duration, False)
        d.set_arm_positions(list(map(float, q)), duration, True)

    def start_streaming(self) -> None:
        if self.read_only:
            return  # dry run: the joints stay in whatever mode they are in
        self._driver.set_arm_modes(self._ta.Mode.position)
        self._driver.set_gripper_mode(self._ta.Mode.position)
        self._streaming = True

    def send(self, q, v, a, gripper_m, goal_time=0.02) -> None:
        if self.read_only:
            return
        if not self._streaming:
            raise RuntimeError("start_streaming() first")
        self._driver.set_arm_positions(list(map(float, q)), goal_time, False, list(map(float, v)), list(map(float, a)))
        self._driver.set_gripper_position(float(gripper_m), goal_time, False)

    def read(self) -> np.ndarray:
        return np.asarray(self._driver.get_arm_positions(), dtype=float)

    def hold(self) -> None:
        if self.read_only or self._driver is None:
            return
        self._driver.set_arm_positions(list(map(float, self.read())), 0.1, False)

    def close(self) -> None:
        if self._driver is not None:
            try:
                self._driver.cleanup()
            finally:
                self._driver = None


class ArmController:
    """Runs the joint pipeline at a fixed rate against one backend.

    ``start()`` connects, checks the arm, and moves it slowly from wherever it is to
    the teleop home pose; ``advance()`` runs the control ticks that are due;
    ``stop()`` holds the arm and returns it to rest. Any ``ArmFault`` raised by a
    tick means the caller should call ``stop()``.
    """

    def __init__(
        self,
        backend,
        kinematics: ArmKinematics,
        smoother: PoseSmoother,
        home: tuple[np.ndarray, np.ndarray],
        *,
        rate_hz: float = 200.0,
        max_velocity: np.ndarray = (1.0, 1.0, 1.0, 2.0, 2.0, 2.0),
        max_acceleration: np.ndarray = (4.0, 4.0, 4.0, 8.0, 8.0, 8.0),
        max_jerk: np.ndarray = (240.0, 240.0, 240.0, 480.0, 480.0, 480.0),
        velocity_fraction: float = 0.5,
        goal_time: float = 0.02,
        move_s: float = 3.0,
        gripper_speed: float = 0.08,
        max_tracking_error: float = 0.12,
        tracking_error_s: float = 0.05,
        sim=None,
    ):
        self.backend = backend
        self.kin = kinematics
        self.ik = ArmIK(kinematics)
        self.smoother = smoother
        self.home = home
        self.dt = 1.0 / rate_hz
        self.max_velocity = np.asarray(max_velocity, dtype=float)
        self.max_acceleration = np.asarray(max_acceleration, dtype=float)
        self.max_jerk = np.asarray(max_jerk, dtype=float)
        self.velocity_fraction = velocity_fraction
        self.goal_time = goal_time
        self.move_s = move_s
        self.gripper_speed = gripper_speed
        self.max_tracking_error = max_tracking_error
        self.tracking_error_ticks = max(1, round(tracking_error_s * rate_hz))
        self.sim = sim  # WidowXSim: physics to step (SimArm) or a display to pose (real arm)
        self.shaper: JointShaper | None = None
        self.q_cmd = REST_Q.copy()
        self.gripper = GRIPPER_OPEN_M
        self.started = False
        self.late_ticks = 0
        self._over_error = 0
        self._t = 0.0
        self._sim_time = 0.0

    @property
    def moves_hardware(self) -> bool:
        return isinstance(self.backend, TrossenArm) and not self.backend.read_only

    def start(self, t0: float = 0.0) -> str:
        state = self.backend.connect()
        v_max = self.max_velocity.copy()
        if state.velocity_max is not None:  # stay well inside what the arm itself allows
            v_max = np.minimum(v_max, self.velocity_fraction * state.velocity_max)
        self.ik.lower = np.maximum(self.ik.lower, state.joint_range[:, 0] + 0.03)
        self.ik.upper = np.minimum(self.ik.upper, state.joint_range[:, 1] - 0.03)
        self.gate_lower, self.gate_upper = state.joint_range[:, 0] + 0.01, state.joint_range[:, 1] - 0.01
        q_home, err_p, err_r = self.ik.solve(*self.home, REST_Q, iterations=300)
        if err_p > 1e-3 or err_r > 1e-2:
            raise ArmFault(f"home pose unreachable (IK residual {err_p * 1000:.1f} mm, {math.degrees(err_r):.1f} deg)")
        if self.backend.read_only:
            q_start = q_home  # nothing moves; commands are computed as if the arm were at home
        else:
            self.backend.move_blocking(q_home, GRIPPER_OPEN_M, self.move_s)
            q_start = self.backend.read()
        self.shaper = JointShaper(q_start, v_max, self.max_acceleration, self.max_jerk,
                                  lower=self.ik.lower, upper=self.ik.upper)
        self.q_cmd = q_start.copy()
        self.q_ik = q_start.copy()
        self.smoother.reset(*self.kin.fk(q_start))
        self.backend.start_streaming()
        self.started = True
        self._t = self._sim_time = t0
        return f"{state.description}; joint speed limits {np.round(v_max, 2)} rad/s"

    def advance(self, cmd: TeleopCommand, now: float) -> int:
        """Run the control tick due by ``now``; returns how many tick periods it covered."""
        due = int((now - self._t) / self.dt)
        if due <= 0:
            return 0
        if due > 3:
            self.late_ticks += 1  # fell behind: one longer tick, never a burst of commands
        self.tick(cmd, min(due, 3) * self.dt)
        self._t += due * self.dt
        if isinstance(self.backend, SimArm):
            while self._sim_time + self.sim.timestep <= self._t:
                self.backend.step()
                self._sim_time += self.sim.timestep
        return due

    def tick(self, cmd: TeleopCommand, dt: float) -> None:
        p, R = self.smoother.step(cmd.position, cmd.rotation, dt)
        # Seeded from the previous solution: the target moves little per tick, so a few
        # iterations track it exactly, and solutions stay on one continuous branch.
        self.q_ik, _, _ = self.ik.solve(p, R, self.q_ik)
        q, v, a = self.shaper.step(self.q_ik, dt)
        self._gate(q, dt)
        g_target = GRIPPER_OPEN_M * float(cmd.gripper)
        self.gripper += float(np.clip(g_target - self.gripper, -self.gripper_speed * dt, self.gripper_speed * dt))
        lead = self.goal_time  # send the point the arm should be at when the goal time ends
        self.backend.send(q + v * lead + 0.5 * a * lead * lead, v, a, self.gripper, lead)
        self.q_cmd = q
        if self.moves_hardware:
            error = float(np.max(np.abs(self.backend.read() - q)))
            self._over_error = self._over_error + 1 if error > self.max_tracking_error else 0
            if self._over_error >= self.tracking_error_ticks:
                raise ArmFault(f"arm is {error:.3f} rad from its command (limit {self.max_tracking_error})")

    def _gate(self, q: np.ndarray, dt: float) -> None:
        """Independent of the shaper: reject anything it should never have produced."""
        if not np.all(np.isfinite(q)):
            raise ArmFault("non-finite joint command")
        outside = np.flatnonzero((q < self.gate_lower) | (q > self.gate_upper))
        if outside.size:
            raise ArmFault(f"joint_{outside[0]} command {q[outside[0]]:.3f} rad is outside its range")
        if np.any(np.abs(q - self.q_cmd) > self.shaper.v_max * dt * 1.05 + 1e-9):
            raise ArmFault("joint command step exceeds the speed limit")

    def measured_q(self) -> np.ndarray:
        return self.q_cmd.copy() if self.backend.read_only else self.backend.read()

    def tracking_error(self) -> float:
        """Distance between the smoothed target and where the gripper tip is (or would be), m."""
        return float(np.linalg.norm(self.smoother.p - self.kin.fk(self.measured_q())[0]))

    def joint_speeds(self) -> np.ndarray:
        if isinstance(self.backend, SimArm):
            return self.sim.joint_speeds()
        return np.abs(self.shaper.v) if self.shaper is not None else np.zeros(6)

    def joint_margins(self) -> np.ndarray:
        q = self.measured_q()
        return np.minimum(q - self.kin.joint_range[:, 0], self.kin.joint_range[:, 1] - q)

    def show(self) -> None:
        """Pose the viewer's arm like the real one (measured) or the dry run's commands."""
        if self.sim is not None and not isinstance(self.backend, SimArm):
            self.sim.data.qpos[self.sim._qpos] = self.measured_q()
            mujoco.mj_forward(self.sim.model, self.sim.data)

    def stop(self) -> None:
        """Hold, then return slowly to rest. Safe to call more than once or after a fault."""
        try:
            if self.started and not self.backend.read_only:
                self.backend.hold()
                time.sleep(0.2)
                self.backend.move_blocking(REST_Q, self.gripper, self.move_s)
        finally:
            self.started = False
            self.backend.close()


def _major_minor(version: str) -> tuple[int, int]:
    parts = str(version).strip().lstrip("v").split(".")
    return int(parts[0]), int(parts[1])


def _package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)
