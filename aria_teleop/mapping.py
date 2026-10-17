"""Teleop logic: gating, filtering, clutch and relative hand-to-robot mapping.

No simulator dependency, so the same mapper can later drive the real WidowX AI.

Mapping (all relative to the moment the clutch engages):
    p_target = p_E0 + scale * A @ (p_wrist - p_wrist0)
    R_target = A @ (R_wrist @ R_wrist0^T) @ A^T @ R_E0
where A rotates room-frame vectors into the robot base frame, built from gravity
and the direction the operator's head faces at engage time.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .eyes import SIDES, EyeState
from .filters import OneEuroFilter, OneEuroRotationFilter
from .fusion import HandFrame, HandState
from .geometry import from_rotvec, heading_frame, interpolate_rotation, rotvec

THUMB_TIP = 0  # Aria landmark indices (projectaria_tools HandLandmark)
INDEX_TIP = 1
OTHER_HAND = {"left": "right", "right": "left"}


@dataclass
class TeleopConfig:
    hand: str = "right"  # hand that drives the arm
    # "key": toggle from the viewer; "pinch": other hand pinched = engaged;
    # "eyes": the viewer key arms, a left wink engages, a right wink releases
    clutch: str = "key"
    scale: float = 1.0  # robot metres per hand metre
    use_rotation: bool = True  # False keeps the gripper orientation fixed
    yaw_offset_deg: float = 0.0  # set to 180 if you face the robot instead of standing behind it
    min_confidence: float = 0.3
    max_hand_speed: float = 3.0  # m/s; faster wrist jumps are treated as tracking glitches
    max_rejects: int = 3  # after this many glitches in a row, accept and re-anchor
    stale_after_s: float = 0.15  # no hand frame for this long counts as lost
    reanchor_after_s: float = 0.3  # lost for this long: re-anchor when the hand returns
    pinch_closed_m: float = 0.02  # thumb-index distance that maps to a closed gripper
    pinch_open_m: float = 0.08  # ... and to a fully open gripper
    clutch_pinch_on_m: float = 0.025  # "pinch" clutch engages below this ...
    clutch_pinch_off_m: float = 0.045  # ... and releases above this
    wink_s: float = 0.6  # "eyes" clutch: one eye closed this long, the other open, is a wink
    eyes_pause_s: float = 0.5  # both eyes closed this long (longer than a blink): robot holds still
    eyes_release_s: float = 2.0  # ... and this long: release the clutch
    eyes_stale_s: float = 0.5  # no eye data for this long while engaged: hold still
    pos_min_cutoff: float = 1.5  # One Euro filter, Hz
    pos_beta: float = 10.0
    rot_min_cutoff: float = 1.5
    rot_beta: float = 2.0
    workspace_min: tuple[float, float, float] = (0.05, -0.45, 0.03)  # robot base frame, m
    workspace_max: tuple[float, float, float] = (0.60, 0.45, 0.55)
    shoulder: tuple[float, float, float] = (0.02, 0.0, 0.10)
    max_reach_m: float = 0.62


class Status(str, Enum):
    IDLE = "idle"  # clutch released: robot holds
    TRACKING = "tracking"  # following the hand
    HOLD = "hold"  # clutch engaged but hand lost: robot holds until it returns
    PAUSED = "paused"  # "eyes" clutch engaged but eyes closed or not tracked: robot holds


@dataclass
class TeleopCommand:
    position: np.ndarray  # robot base frame, m
    rotation: np.ndarray  # 3x3, robot base frame
    gripper: float  # openness: 0 closed, 1 open
    status: Status


@dataclass
class _Anchor:
    wrist_position: np.ndarray
    wrist_rotation: np.ndarray
    target_position: np.ndarray
    target_rotation: np.ndarray
    A: np.ndarray
    session: object


def pinch_distance(hand: HandState | None) -> float | None:
    if hand is None or hand.landmarks_room is None or len(hand.landmarks_room) <= INDEX_TIP:
        return None
    return float(np.linalg.norm(hand.landmarks_room[THUMB_TIP] - hand.landmarks_room[INDEX_TIP]))


def clamp_to_workspace(p: np.ndarray, cfg: TeleopConfig) -> np.ndarray:
    p = np.clip(p, cfg.workspace_min, cfg.workspace_max)
    shoulder = np.asarray(cfg.shoulder)
    v = p - shoulder
    r = float(np.linalg.norm(v))
    if r > cfg.max_reach_m:
        p = shoulder + v * (cfg.max_reach_m / r)
    return p


class TeleopMapper:
    """Turns room-frame hand frames into robot end-effector targets."""

    def __init__(self, config: TeleopConfig, home_position: np.ndarray, home_rotation: np.ndarray):
        self.cfg = config
        self._target_p = clamp_to_workspace(np.array(home_position, dtype=float), config)
        self._target_R = np.array(home_rotation, dtype=float)
        self._gripper = 1.0
        self._key_engaged = False
        self._pinch_engaged = False
        self._clutch_hand_seen_at = -math.inf
        self._armed = False  # "eyes" clutch: winks only count while armed
        self._eyes_engaged = False
        # A wink fires once; the eye has to reopen before it can fire again.
        self._wink_spent = {side: False for side in SIDES}
        self._anchor: _Anchor | None = None
        # Operator heading, taken from the head at the first engage and kept for
        # the VIO session so re-clutching doesn't rotate the mapping with head sway.
        self._heading: tuple[np.ndarray, object] | None = None
        self._last_t_ns: int | None = None
        self._last_raw_p: np.ndarray | None = None
        self._last_raw_t_ns: int | None = None
        self._last_session: object = None
        self._rejects = 0
        self._last_good_at = -math.inf
        self._pos_filter = OneEuroFilter(config.pos_min_cutoff, config.pos_beta)
        self._rot_filter = OneEuroRotationFilter(config.rot_min_cutoff, config.rot_beta)
        self._grip_filter = OneEuroFilter(min_cutoff=3.0, beta=0.0)
        self.status = Status.IDLE
        self.rejected_frames = 0
        self.anchors = 0
        self.last_confidence: float | None = None
        self.last_pinch: float | None = None

    # Clutch input. Called from the viewer's key callback thread; plain bool writes.
    def toggle_clutch(self) -> None:
        if self.cfg.clutch == "eyes":
            self._armed = not self._armed  # disarming also releases (see _update_eye_clutch)
        else:
            self._key_engaged = not self._key_engaged

    def set_clutch(self, engaged: bool) -> None:
        if self.cfg.clutch == "eyes":
            self._armed = self._eyes_engaged = engaged
        else:
            self._key_engaged = engaged

    def reset_heading(self) -> None:
        """Re-take "forward" from where the head faces at the next engage."""
        self._heading = None
        self._anchor = None

    @property
    def armed(self) -> bool:
        """Only meaningful for the "eyes" clutch."""
        return self.cfg.clutch == "eyes" and self._armed

    @property
    def engaged(self) -> bool:
        if self.cfg.clutch == "pinch":
            return self._pinch_engaged
        if self.cfg.clutch == "eyes":
            return self._armed and self._eyes_engaged
        return self._key_engaged

    def update(self, frame: HandFrame | None, now: float, eyes: EyeState | None = None) -> TeleopCommand:
        cfg = self.cfg
        fresh = frame is not None and now - frame.received_at <= cfg.stale_after_s
        hand = None
        if fresh and frame.t_ns != self._last_t_ns:
            self._last_t_ns = frame.t_ns
            self._update_pinch_clutch(frame, now)
            hand = self._accept(frame)
        elif not fresh:
            self._pinch_engaged = False  # no data at all: release the deadman
        paused = self._update_eye_clutch(eyes, now)

        if not self.engaged:
            self._anchor = None
            self._pos_filter.reset()
            self._rot_filter.reset()
            self.status = Status.IDLE
            return self._command()

        if paused:
            # Not looking: hold still, and re-anchor once the eyes open so hand
            # motion made while the operator couldn't see is ignored.
            self._anchor = None
            self._pos_filter.reset()
            self._rot_filter.reset()
            self.status = Status.PAUSED
            return self._command()

        if hand is not None:
            self._track(hand, frame, now)
        if now - self._last_good_at > cfg.reanchor_after_s and self._anchor is not None:
            # Lost long enough that motion during the gap is unknown: start over
            # from wherever the robot is when the hand comes back.
            self._anchor = None
            self._pos_filter.reset()
            self._rot_filter.reset()
        tracking = self._anchor is not None and now - self._last_good_at <= cfg.stale_after_s
        self.status = Status.TRACKING if tracking else Status.HOLD
        return self._command()

    def _command(self) -> TeleopCommand:
        return TeleopCommand(self._target_p.copy(), self._target_R.copy(), self._gripper, self.status)

    def _update_pinch_clutch(self, frame: HandFrame, now: float) -> None:
        cfg = self.cfg
        if cfg.clutch != "pinch":
            return
        other = frame.hands.get(OTHER_HAND[cfg.hand])
        if other is not None and other.confidence < cfg.min_confidence:
            other = None
        d = pinch_distance(other)
        if d is None:
            # Ride out single-frame dropouts of the clutch hand, release after that.
            if now - self._clutch_hand_seen_at > cfg.stale_after_s:
                self._pinch_engaged = False
            return
        self._clutch_hand_seen_at = now
        if d < cfg.clutch_pinch_on_m:
            self._pinch_engaged = True
        elif d > cfg.clutch_pinch_off_m:
            self._pinch_engaged = False

    def _update_eye_clutch(self, eyes: EyeState | None, now: float) -> bool:
        """Apply wink commands; returns True while the robot must hold for the eyes."""
        cfg = self.cfg
        if cfg.clutch != "eyes":
            return False
        if not self._armed:
            self._eyes_engaged = False
            return False
        if eyes is None or now - eyes.received_at > cfg.eyes_stale_s:
            return True  # eye tracking lost: can't tell whether the operator is looking
        closed = {"left": eyes.left_closed_s, "right": eyes.right_closed_s}
        for side in SIDES:
            if closed[side] == 0.0:
                self._wink_spent[side] = False
        if eyes.both_closed_s >= cfg.eyes_pause_s:
            # A real both-eyes closure, not a blink: when one eye opens first, the
            # other one still being closed must not count as a wink.
            self._wink_spent = {side: True for side in SIDES}
            if eyes.both_closed_s >= cfg.eyes_release_s:
                self._eyes_engaged = False
            return True
        for side, other in (("left", "right"), ("right", "left")):
            if closed[side] >= cfg.wink_s and closed[other] == 0.0 and not self._wink_spent[side]:
                self._wink_spent[side] = True
                self._eyes_engaged = side == "left"  # left wink engages, right wink releases
        return False

    def _accept(self, frame: HandFrame) -> HandState | None:
        """Gate one frame of the driving hand: missing, low confidence, or a glitch."""
        cfg = self.cfg
        hand = frame.hands.get(cfg.hand)
        if hand is None:
            return None
        self.last_confidence = hand.confidence
        if hand.confidence < cfg.min_confidence:
            return None
        p = hand.T_room_wrist[:3, 3]
        if self._last_raw_p is not None and frame.session == self._last_session:
            dt = max((frame.t_ns - self._last_raw_t_ns) * 1e-9, 1e-3)
            speed = float(np.linalg.norm(p - self._last_raw_p)) / dt
            if speed > cfg.max_hand_speed:
                if self._rejects < cfg.max_rejects:
                    self._rejects += 1
                    self.rejected_frames += 1
                    return None
                # A sustained jump is a real discontinuity (e.g. VIO relocalised).
                self._anchor = None
                self._pos_filter.reset()
                self._rot_filter.reset()
        self._rejects = 0
        self._last_raw_p = p.copy()
        self._last_raw_t_ns = frame.t_ns
        self._last_session = frame.session
        return hand

    def _track(self, hand: HandState, frame: HandFrame, now: float) -> None:
        cfg = self.cfg
        t = frame.t_ns * 1e-9
        p = self._pos_filter(hand.T_room_wrist[:3, 3], t)
        R = self._rot_filter(hand.T_room_wrist[:3, :3], t)
        if self._anchor is None or self._anchor.session != frame.session:
            if self._heading is None or self._heading[1] != frame.session:
                A = heading_frame(
                    frame.T_room_head[:3, 2], -frame.gravity_room, math.radians(cfg.yaw_offset_deg)
                )
                if A is None:
                    return  # looking straight down/up: heading undefined, wait for a usable frame
                self._heading = (A, frame.session)
            A = self._heading[0]
            self._anchor = _Anchor(p, R, self._target_p.copy(), self._target_R.copy(), A, frame.session)
            self.anchors += 1
        self._last_good_at = now
        a = self._anchor
        self._target_p = clamp_to_workspace(
            a.target_position + cfg.scale * a.A @ (p - a.wrist_position), cfg
        )
        if cfg.use_rotation:
            self._target_R = a.A @ (R @ a.wrist_rotation.T) @ a.A.T @ a.target_rotation
        d = pinch_distance(hand)
        if d is not None:
            self.last_pinch = d
            openness = (d - cfg.pinch_closed_m) / (cfg.pinch_open_m - cfg.pinch_closed_m)
            self._gripper = float(self._grip_filter(np.array([np.clip(openness, 0.0, 1.0)]), t)[0])


class PoseSmoother:
    """Turns ~30 Hz target updates into a smooth high-rate command.

    Each new goal is reached by linear interpolation over ``blend_time`` (about
    one hand-tracking period) instead of a step; a speed cap keeps the arm safe.
    """

    def __init__(self, blend_time: float = 0.035, max_speed: float = 0.6, max_angular_speed: float = 4.0):
        self.blend_time = blend_time
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.reset(np.zeros(3), np.eye(3))

    def reset(self, p: np.ndarray, R: np.ndarray) -> None:
        self.p, self.R = np.array(p, dtype=float), np.array(R, dtype=float)
        self._start_p, self._start_R = self.p.copy(), self.R.copy()
        self._goal_p, self._goal_R = self.p.copy(), self.R.copy()
        self._elapsed = self.blend_time

    def step(self, p_goal: np.ndarray, R_goal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
        if not (np.array_equal(p_goal, self._goal_p) and np.array_equal(R_goal, self._goal_R)):
            self._start_p, self._start_R = self.p.copy(), self.R.copy()
            self._goal_p, self._goal_R = np.array(p_goal, dtype=float), np.array(R_goal, dtype=float)
            self._elapsed = 0.0
        self._elapsed = min(self._elapsed + dt, self.blend_time)
        a = self._elapsed / self.blend_time
        p_des = self._start_p + a * (self._goal_p - self._start_p)
        R_des = interpolate_rotation(self._start_R, self._goal_R, a)

        d = p_des - self.p
        n = float(np.linalg.norm(d))
        if n > self.max_speed * dt:
            d *= self.max_speed * dt / n
        self.p = self.p + d
        r = rotvec(self.R.T @ R_des)
        ang = float(np.linalg.norm(r))
        if ang > self.max_angular_speed * dt:
            r *= self.max_angular_speed * dt / ang
        self.R = self.R @ from_rotvec(r)
        return self.p.copy(), self.R.copy()
