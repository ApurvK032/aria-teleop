"""MuJoCo backend: drives Trossen's WidowX AI model from ``TeleopCommand``s.

Uses the follow-target scene and damped-least-squares IK controller from
``trossen_arm_mujoco``. The scene's blue cube shows the commanded target and
changes colour with the teleop status.
"""

from __future__ import annotations

from contextlib import nullcontext
from importlib.resources import files
import time
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np
from trossen_arm_mujoco.src.controller import Controller, RobotType

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
CLUTCH_MESSAGES = {  # clutch state -> (terminal line, words spoken on the glasses)
    "engaged": ("clutch ENGAGED", "engaged"),
    "paused": ("clutch paused: eyes closed or not tracked", "paused"),
    "released": ("clutch released", "released"),
    "armed": ("armed: wink your left eye to engage, your right eye to release", "armed"),
    "disarmed": ("disarmed", "disarmed"),
}


def clutch_state(mapper: TeleopMapper, status: Status) -> str:
    if mapper.engaged:
        return "paused" if status is Status.PAUSED else "engaged"
    if mapper.cfg.clutch == "eyes":
        return "armed" if mapper.armed else "disarmed"
    return "released"


class WidowXSim:
    def __init__(self, smoother: PoseSmoother):
        self.model = mujoco.MjModel.from_xml_path(str(files("trossen_arm_mujoco") / SCENE))
        self.data = mujoco.MjData(self.model)
        self.robot = Controller(self.model, self.data, robot_type=RobotType.WXAI, ik_scale=1.0, ik_damping=0.03)
        self._cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_cube_visual")
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
        self.robot.set_ee_pose(p, quat_wxyz(R))
        lo, hi = GRIPPER_RANGE
        self.robot.set_gripper_position(lo + cmd.gripper * (hi - lo))
        mujoco.mj_step(self.model, self.data)

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
) -> dict:
    """Real-time loop: map the latest hand frame, step physics to wall time, render.

    Clutch changes are printed, and passed to ``announce`` (speech on the glasses).
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
    stats = {"steps": 0, "max_error_m": 0.0, "error_sum_m": 0.0, "tracking_steps": 0, "behind": 0}
    with viewer_ctx as viewer:
        if viewer is not None:
            viewer.cam.lookat[:] = [0.3, 0.0, 0.15]
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
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
                print(line)
                if announce is not None:
                    announce(spoken)
                last_clutch = state
            if hand_viewer is not None:
                hand_viewer.log(frame, now, eyes)
            steps = 0
            while sim_time < elapsed and steps < MAX_CATCHUP_STEPS:
                sim.step(cmd)
                sim_time += sim.timestep
                steps += 1
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
                x, y, z = cmd.position
                print(
                    f"[{elapsed:6.1f}s] {cmd.status.value:8s} clutch={clutch} "
                    f"hands {hand_hz:4.1f}Hz head {head_hz:5.0f}Hz{eye_rate} conf {conf} pinch {pinch} "
                    f"target ({x:+.3f},{y:+.3f},{z:+.3f}) err {sim.tracking_error() * 1000:5.1f}mm "
                    f"rejected {mapper.rejected_frames} anchors {mapper.anchors}"
                )
            time.sleep(0.001)

    if stats["tracking_steps"]:
        stats["mean_error_m"] = stats["error_sum_m"] / stats["tracking_steps"]
    return stats
