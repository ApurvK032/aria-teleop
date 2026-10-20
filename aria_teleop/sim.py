"""MuJoCo backend: drives Trossen's WidowX AI model from ``TeleopCommand``s.

Uses the follow-target scene and damped-least-squares IK controller from
``trossen_arm_mujoco``. The scene's blue cube shows the commanded target and
changes colour with the teleop status.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from importlib.resources import files
import os
import select
import sys
import threading
import time
from typing import Callable, Sequence

import mujoco
import mujoco.viewer
import numpy as np
from trossen_arm_mujoco.src.controller import Controller, RobotType

from .arm import ArmController
from .fusion import HandFuser
from .geometry import quat_wxyz
from .hand_viewer import HandViewer
from .mapping import PoseSmoother, Status, TeleopCommand, TeleopMapper

SCENE = "assets/wxai/scene_wxai_follow_target.xml"
HOME_POSITION = np.array([0.3, 0.0, 0.2])  # same start target as Trossen's follow-target demo
HOME_ROTATION = np.eye(3)
GRIPPER_RANGE = (0.0, 0.044)  # carriage travel in the sim model, m
STATUS_COLORS = {
    Status.IDLE: (0.6, 0.6, 0.6, 0.5),
    Status.TRACKING: (0.1, 0.8, 0.2, 0.6),
    Status.HOLD: (1.0, 0.55, 0.0, 0.7),
    Status.PAUSED: (0.25, 0.45, 1.0, 0.7),
}
MAX_CATCHUP_STEPS = 50  # if the PC falls behind real time, drop the backlog instead of spiralling
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist pitch", "wrist yaw", "wrist roll")  # joint_0 .. joint_5
CLUTCH_MESSAGES = {  # clutch state -> (terminal line, words spoken on the glasses)
    "engaged": ("clutch ENGAGED", "engaged"),
    "paused": ("clutch paused: eyes closed or not tracked", "paused"),
    "released": ("clutch released", "released"),
    "armed": ("armed: wink your left eye to engage, your right eye to release", "armed"),
    "disarmed": ("disarmed", "disarmed"),
}


@contextmanager
def terminal_keys(on_key: Callable[[int], None]):
    """Pass single key presses in the terminal to ``on_key``, so SPACE and H work without the viewer.

    The terminal goes into cbreak mode (no line buffering or echo; Ctrl+C still interrupts)
    and is restored on exit. Does nothing when stdin isn't a terminal.
    """
    if not sys.stdin.isatty():
        yield
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    stop = threading.Event()

    def read_keys() -> None:
        while not stop.is_set():
            if select.select([fd], [], [], 0.1)[0]:
                key = os.read(fd, 1)
                if key:
                    on_key(ord(chr(key[0]).upper()))  # same codes as the viewer: SPACE 32, H 72

    tty.setcbreak(fd)
    thread = threading.Thread(target=read_keys, daemon=True)
    thread.start()
    print("terminal keys: SPACE toggles the clutch (arms with --clutch eyes), H resets heading, Ctrl+C stops")
    try:
        yield
    finally:
        stop.set()
        thread.join(0.5)
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def clutch_state(mapper: TeleopMapper, status: Status) -> str:
    if mapper.engaged:
        return "paused" if status is Status.PAUSED else "engaged"
    if mapper.cfg.clutch == "eyes":
        return "armed" if mapper.armed else "disarmed"
    return "released"


class WidowXSim:
    """The simulated arm. The simulator lets joints move as fast as the IK asks; the real arm
    has per-joint speed limits (read them with ``TrossenArmDriver.get_joint_limits()``).
    ``joint_speed_limit`` (rad/s, one value or one per joint) sets what ``run`` warns about,
    and ``cap_joint_speed`` also holds the IK to it, so the sim moves the way the real arm could.
    """

    def __init__(
        self,
        smoother: PoseSmoother,
        joint_speed_limit: float | Sequence[float] | None = None,
        cap_joint_speed: bool = False,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(files("trossen_arm_mujoco") / SCENE))
        self.data = mujoco.MjData(self.model)
        self.robot = Controller(self.model, self.data, robot_type=RobotType.WXAI, ik_scale=1.0, ik_damping=0.03)
        self._cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube_visual")
        self._qpos = np.array(self.robot.arm_qpos_indices)
        self._dofs = np.array(self.robot.arm_dof_indices)
        self.joint_range = self.model.jnt_range[self.robot.arm_joint_ids].copy()  # (6, 2), rad
        self.joint_speed_limit = (
            None if joint_speed_limit is None
            else np.broadcast_to(np.asarray(joint_speed_limit, dtype=float), (len(self._dofs),)).copy()
        )
        if cap_joint_speed and self.joint_speed_limit is None:
            raise ValueError("cap_joint_speed needs joint_speed_limit")
        self.cap_joint_speed = cap_joint_speed
        self._q_ref = self.data.qpos[self._qpos].copy()  # joint reference sent to the servos
        mujoco.mj_forward(self.model, self.data)
        self.smoother = smoother
        p, R = self.ee_pose()
        self.smoother.reset(p, R)  # start from where the arm is; the smoother walks it to home

    @property
    def timestep(self) -> float:
        return self.model.opt.timestep

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        site = self.data.site(self.robot.ee_site_id)
        return site.xpos.copy(), site.xmat.reshape(3, 3).copy()

    def step(self, cmd: TeleopCommand) -> None:
        p, R = self.smoother.step(cmd.position, cmd.rotation, self.timestep)
        # Where the IK wants the joints this step (what Controller.set_ee_pose applies).
        q_goal = self.data.qpos[self._qpos] + self.robot.compute_ik_step(p, quat_wxyz(R))
        if self.cap_joint_speed:
            # Rate-limit the joint reference itself, like a position-mode arm with velocity
            # limits. (Clipping the IK step instead would stall the servos: it re-bases on the
            # current angle every step, so the servo never gets enough lead to move.)
            # 80%: the servos overshoot a rate-limited reference by up to ~15%, so this keeps the
            # measured speed under the limit that run() warns about.
            step_max = 0.8 * self.joint_speed_limit * self.timestep
            q_goal = np.clip(q_goal, self._q_ref - step_max, self._q_ref + step_max)
        self._q_ref = q_goal
        self.robot.set_arm_joint_positions(q_goal)
        lo, hi = GRIPPER_RANGE
        self.robot.set_gripper_position(lo + cmd.gripper * (hi - lo))
        mujoco.mj_step(self.model, self.data)

    def joint_speeds(self) -> np.ndarray:
        return np.abs(self.data.qvel[self._dofs])

    def joint_margins(self) -> np.ndarray:
        """How far each arm joint is from the nearer end of its range, rad."""
        q = self.data.qpos[self._qpos]
        return np.minimum(q - self.joint_range[:, 0], self.joint_range[:, 1] - q)

    def tracking_error(self) -> float:
        """Distance between the smoothed command and the actual gripper tip, m."""
        return float(np.linalg.norm(self.smoother.p - self.ee_pose()[0]))

    def show_target(self, cmd: TeleopCommand) -> None:
        self.data.mocap_pos[0] = cmd.position
        self.data.mocap_quat[0] = quat_wxyz(cmd.rotation)
        self.model.geom_rgba[self._cube_geom] = STATUS_COLORS[cmd.status]


def run(
    sim: WidowXSim,
    mapper: TeleopMapper,
    fuser: HandFuser,
    *,
    headless: bool = False,
    duration: float | None = None,
    auto_engage_after: float | None = None,
    clutch_key: int = ord(" "),
    status_period: float = 1.0,
    hand_viewer: HandViewer | None = None,
    announce: Callable[[str], None] | None = None,
    joint_limit_margin_deg: float = 5.0,
    arm: ArmController | None = None,
) -> dict:
    """Real-time loop: map the latest hand frame, step physics to wall time, render.

    Clutch changes are printed, and passed to ``announce`` (speech on the glasses).
    While the clutch drives the arm, joints faster than ``sim.joint_speed_limit`` or
    within ``joint_limit_margin_deg`` of their range end are warned about, once a second.

    With ``arm``, joint commands come from its fixed-rate pipeline instead of the
    simulator's own IK: it drives the MuJoCo arm (preview) or the real arm, and the
    viewer shows what the arm is doing. An ``ArmFault`` propagates to the caller,
    which must call ``arm.stop()``.
    """

    def on_key(keycode: int) -> None:
        if keycode == clutch_key:
            mapper.toggle_clutch()  # the loop reports the change
        elif keycode == ord("H"):
            mapper.reset_heading()
            print("heading reset: 'forward' = where you face at the next engage")

    viewer_ctx = (
        nullcontext(None) if headless else mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=on_key)
    )
    stats = {"steps": 0, "max_error_m": 0.0, "error_sum_m": 0.0, "tracking_steps": 0, "behind": 0,
             "joint_speed_peak": np.zeros(len(JOINT_NAMES)), "speed_warnings": 0, "limit_warnings": 0}
    margin_rad = np.radians(joint_limit_margin_deg)
    win_speed = np.zeros(len(JOINT_NAMES))  # peak joint speeds and closest range margins while
    win_margin = np.full(len(JOINT_NAMES), np.inf)  # driving the arm, since the last status line
    with viewer_ctx as viewer, terminal_keys(on_key):
        if viewer is not None:
            viewer.cam.lookat[:] = [0.3, 0.0, 0.15]
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
        tracking_error = sim.tracking_error
        if arm is not None:
            doing = "connecting (no motion)" if arm.backend.read_only else "connecting, then moving slowly to home"
            print(f"arm ({arm.backend.name}): {doing}... ", end="", flush=True)
            print(arm.start())
            tracking_error = arm.tracking_error
        start = time.monotonic()
        sim_time = 0.0
        last_sync = last_status = start
        last_hand_frames, last_head_poses, last_eye_samples = fuser.hand_frames, fuser.head_poses, fuser.eyes.samples
        last_clutch = clutch_state(mapper, mapper.status)
        while viewer is None or viewer.is_running():
            now = time.monotonic()
            elapsed = now - start
            if duration is not None and elapsed >= duration:
                break
            if auto_engage_after is not None and elapsed >= auto_engage_after:
                mapper.set_clutch(True)
                auto_engage_after = None
                print("(auto-engage)")

            frame, eyes = fuser.latest(), fuser.eyes.state()
            cmd = mapper.update(frame, now, eyes)
            state = clutch_state(mapper, cmd.status)
            if state != last_clutch:
                line, spoken = CLUTCH_MESSAGES[state]
                if state == "armed" and last_clutch in ("engaged", "paused"):
                    line, spoken = "clutch released (still armed)", "released"
                if mapper.cfg.clutch == "eyes" and eyes is not None:  # what triggered it
                    line += (f"  [eyes closed L{eyes.left_closed_s:.2f} R{eyes.right_closed_s:.2f} "
                             f"both{eyes.both_closed_s:.2f} s]")
                print(f"[{elapsed:6.1f}s] {line}")
                if announce is not None:
                    announce(spoken)
                last_clutch = state
            if hand_viewer is not None:
                hand_viewer.log(frame, now, eyes)
            steps = 0
            driving = cmd.status is not Status.IDLE
            if arm is not None and arm.advance(cmd, elapsed):
                if driving:
                    np.maximum(win_speed, arm.joint_speeds(), out=win_speed)
                    np.minimum(win_margin, arm.joint_margins(), out=win_margin)
                if cmd.status is Status.TRACKING:
                    err = arm.tracking_error()
                    stats["error_sum_m"] += err
                    stats["max_error_m"] = max(stats["max_error_m"], err)
                    stats["tracking_steps"] += 1
            while arm is None and sim_time < elapsed and steps < MAX_CATCHUP_STEPS:
                sim.step(cmd)
                sim_time += sim.timestep
                steps += 1
                if driving:
                    np.maximum(win_speed, sim.joint_speeds(), out=win_speed)
                    np.minimum(win_margin, sim.joint_margins(), out=win_margin)
                if cmd.status is Status.TRACKING:
                    err = sim.tracking_error()
                    stats["error_sum_m"] += err
                    stats["max_error_m"] = max(stats["max_error_m"], err)
                    stats["tracking_steps"] += 1
            stats["steps"] += steps
            if steps == MAX_CATCHUP_STEPS:
                sim_time = elapsed
                stats["behind"] += 1

            if viewer is not None and now - last_sync >= 1 / 60:
                sim.show_target(cmd)
                if arm is not None:
                    arm.show()
                viewer.sync()
                last_sync = now
            if now - last_status >= status_period:
                period = now - last_status
                hand_hz = (fuser.hand_frames - last_hand_frames) / period
                head_hz = (fuser.head_poses - last_head_poses) / period
                eye_hz = (fuser.eyes.samples - last_eye_samples) / period
                last_hand_frames, last_head_poses = fuser.hand_frames, fuser.head_poses
                last_eye_samples = fuser.eyes.samples
                last_status = now
                conf = "  -  " if mapper.last_confidence is None else f"{mapper.last_confidence:.2f}"
                pinch = "  -  " if mapper.last_pinch is None else f"{mapper.last_pinch * 100:4.1f}cm"
                clutch = "on " if mapper.engaged else "arm" if mapper.armed else "off"
                eye_rate = f" eyes {eye_hz:4.1f}Hz" if mapper.cfg.clutch == "eyes" else ""
                joints = " joints   -"
                if np.isfinite(win_margin).all():  # the arm was driven during this period
                    limit = sim.joint_speed_limit
                    over = [] if limit is None else np.flatnonzero(win_speed > limit)
                    near = np.flatnonzero(win_margin < margin_rad)
                    warnings = [f"{JOINT_NAMES[j]} {win_speed[j]:.2f} rad/s > {limit[j]:.2f}" for j in over]
                    warnings += [f"{JOINT_NAMES[j]} {np.degrees(win_margin[j]):.0f} deg from its range end" for j in near]
                    if warnings:
                        print(f"[{elapsed:6.1f}s] WARNING joints: " + "; ".join(warnings))
                    stats["speed_warnings"] += len(over) > 0
                    stats["limit_warnings"] += len(near) > 0
                    j = int(np.argmin(win_margin))
                    joints = (f" joints {win_speed[:3].max():4.2f}/{win_speed[3:].max():4.2f}rad/s "
                              f"limit j{j} {np.degrees(win_margin[j]):3.0f}deg")
                    np.maximum(stats["joint_speed_peak"], win_speed, out=stats["joint_speed_peak"])
                win_speed[:] = 0.0
                win_margin[:] = np.inf
                x, y, z = cmd.position
                print(
                    f"[{elapsed:6.1f}s] {cmd.status.value:8s} clutch={clutch} "
                    f"hands {hand_hz:4.1f}Hz head {head_hz:5.0f}Hz{eye_rate} conf {conf} pinch {pinch} "
                    f"target ({x:+.3f},{y:+.3f},{z:+.3f}) err {tracking_error() * 1000:5.1f}mm{joints} "
                    f"rejected {mapper.rejected_frames} anchors {mapper.anchors}"
                )
            time.sleep(0.001)

    np.maximum(stats["joint_speed_peak"], win_speed, out=stats["joint_speed_peak"])  # the last partial period
    if arm is not None:
        stats["behind"] = arm.late_ticks
    if stats["tracking_steps"]:
        stats["mean_error_m"] = stats["error_sum_m"] / stats["tracking_steps"]
    return stats
